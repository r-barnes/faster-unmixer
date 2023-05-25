#!/usr/bin/env python3

import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

# TODO(rbarnes): Make a requirements file for conda
import cvxpy as cp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import pyfastunmix

NO_DOWNSTREAM: Final[int] = 0
SAMPLE_CODE_COL_NAME: Final[str] = "Sample.Code"
ELEMENT_LIST: Final[List[str]] = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]  # fmt: skip

ElementData = Dict[str, float]
ExportRateData = Dict[str, float]


class ReciprocalParameter:
    """Used for times when you want a cvxpy Parameter and its ratio"""

    def __init__(self, *args, **kwargs) -> None:
        self._p = cp.Parameter(*args, **kwargs)
        # Reciprocal of the above
        self._rp = cp.Parameter(*args, **kwargs)

    @property
    def value(self) -> Optional[float]:
        """Return the value of the Parameter"""
        return self._p.value

    @value.setter
    def value(self, val: Optional[float]) -> None:
        """
        Simultaneously set the value of the Parameter (given by `p`)
        and its reciprocal (given by `rp`)
        """
        self._p.value = val
        self._rp.value = 1 / val if val is not None else None

    @property
    def p(self) -> cp.Parameter:
        """Returns the parameter"""
        return self._p

    @property
    def rp(self) -> cp.Parameter:
        """Returns the reciprocal of the parameter"""
        return self._rp


def cp_log_ratio(a, b: ReciprocalParameter):
    return cp.maximum(a * b.rp, b.p * cp.inv_pos(a))


def geo_mean(x: List[float]) -> float:
    """Returns geometric mean of a list"""
    return np.exp(np.log(x).mean())


def nx_topological_sort_with_data(
    G: nx.DiGraph,
) -> Iterator[Tuple[str, pyfastunmix.SampleNode]]:
    return ((x, G.nodes[x]["data"]) for x in nx.topological_sort(G))


def nx_get_downstream(G: nx.DiGraph, x: str) -> Optional[str]:
    """Gets the downstream child from a node with only one child"""
    s: List[str] = list(G.successors(x))
    if len(s) == 0:
        return None
    elif len(s) == 1:
        return s[0]
    else:
        raise Exception("More than one downstream neighbour!")


def plot_network(G: nx.DiGraph) -> None:
    ag = nx.nx_agraph.to_agraph(G)
    ag.layout(prog="dot")
    temp = tempfile.NamedTemporaryFile(delete=False)
    tempname = temp.name + ".png"
    ag.draw(tempname)
    img = mpimg.imread(tempname)
    plt.imshow(img)
    plt.show()
    os.remove(tempname)


def get_sample_graphs(
    flowdirs_filename: str,
    sample_data_filename: str,
) -> Tuple[nx.DiGraph, "pyfastunmix.SampleAdjacency"]:
    # Get the graph representations of the data
    sample_network_raw, sample_adjacency = pyfastunmix.fastunmix(
        flowdirs_filename, sample_data_filename
    )

    # Convert it into a networkx graph for easy use in Python
    sample_network = nx.DiGraph()
    for x in sample_network_raw.values():  # Skip the first node into which it all flows
        if x.name == pyfastunmix.root_node_name:
            continue
        sample_network.add_node(x.name, data=x)
        if x.downstream_node != pyfastunmix.root_node_name:
            sample_network.add_edge(x.name, x.downstream_node)

    return sample_network, sample_adjacency


class SampleNetwork:
    # TODO Docstrings
    def __init__(
        self,
        sample_network: nx.DiGraph,
        sample_adjacency: Optional["pyfastunmix.SampleAdjacency"] = None,
        use_regularization: bool = True,
        continuous: bool = False,
        area_labels: Optional[np.array] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
    ) -> None:
        self.sample_network = sample_network
        self.sample_adjacency = sample_adjacency
        self.continuous: bool = continuous
        if self.continuous:
            self.grid = InverseGrid(nx, ny, area_labels, sample_network)
        self._site_to_observation: Dict[str, ReciprocalParameter] = {}
        self._site_to_export_rate: Dict[str, cp.Parameter] = {}
        self._site_to_inverse_total_flux: Dict[str, cp.Parameter] = {}
        self._primary_terms = []
        self._regularizer_terms = []
        self._constraints = []
        self._regularizer_strength = cp.Parameter(nonneg=True)
        self._problem = None
        self._build_primary_terms()
        if use_regularization:
            if continuous:
                self._build_regularizer_terms_continuous()
            else:
                self._build_regularizer_terms_discrete()
        self._build_problem()

    def _build_primary_terms(self) -> None:
        for _, data in self.sample_network.nodes(data=True):
            data["data"].my_total_tracer_flux = 0.0
            data["data"].my_total_flux = 0.0

        # Build the main objective
        # Use a topological sort to ensure an upstream-to-downstream traversal
        for sample_name, my_data in nx_topological_sort_with_data(self.sample_network):
            # Set up a CVXPY parameter for each element for each node
            if self.continuous:
                concs = [node.concentration for node in self.grid.sites_to_nodes[sample_name]]

                my_data.my_tracer_value = cp.sum(concs) / len(
                    concs
                )  # mean conc of all inversion nodes upstream
            else:
                my_data.my_tracer_value = cp.Variable(pos=True)

            # Export rate of total material (e.g., erosion rate, run-off)
            my_data.my_export_rate = cp.Parameter(pos=True)
            self._site_to_export_rate[my_data.name] = my_data.my_export_rate

            # Area weighted total contribution of material from this node
            my_data.my_flux = my_data.area * my_data.my_export_rate
            # Add the flux I generate to the total flux passing through me
            my_data.my_total_flux += my_data.my_flux
            # Area weighted contribution of *tracer* from this node
            my_data.my_tracer_flux = my_data.my_flux * my_data.my_tracer_value
            # Add the *tracer* flux I generate to the total flux of *tracer* passing through me
            my_data.my_total_tracer_flux += my_data.my_tracer_flux

            # total_flux for a given parameter set is variable free but its inverse is not DPP.
            # To make it DPP we set up a dummy parameter for the inverse. We set the value of this paramter
            # at solve time, alongside the export rate and observation parameters.
            inverse_total_flux_dummy = cp.Parameter(pos=True)
            self._site_to_inverse_total_flux[my_data.name] = inverse_total_flux_dummy

            # Set up a dummy (parameter free) variable that encodes the total *tracer* flux at the node.
            # This ensures that the problem is DPP.
            total_tracer_flux_dummy = cp.Variable(pos=True)
            # We add a constraint that this must equal the parameter encoded `total_tracer_flux`
            self._constraints.append(total_tracer_flux_dummy == my_data.my_total_tracer_flux)

            # Set up a dummy (parameter free) variable for normalised concentration.
            # This ensures that the problem is DPP.
            normalised_concentration = total_tracer_flux_dummy * inverse_total_flux_dummy
            normalised_concentration_dummy = cp.Variable(pos=True)
            # We add a constraint that this must equal the parameter encoded `normalised_concentration`
            self._constraints.append(normalised_concentration_dummy == normalised_concentration)

            # Set up a parameter for the observation at node
            observed = ReciprocalParameter(pos=True)
            self._site_to_observation[my_data.name] = observed

            # Calculate misfit and append to primary terms in objective function
            misfit = cp_log_ratio(normalised_concentration_dummy, observed)
            self._primary_terms.append(misfit)

            if ds := nx_get_downstream(self.sample_network, sample_name):
                downstream_data = self.sample_network.nodes[ds]["data"]
                # Add our flux to downstream node's
                downstream_data.my_total_flux += my_data.my_total_flux
                # Add our *tracer* flux to the downstream node's
                downstream_data.my_total_tracer_flux += my_data.my_total_tracer_flux

    def _build_regularizer_terms_continuous(self) -> None:
        # Build the regularizer
        # Loop through all nodes in grid
        for node in self.grid.node_arr.flatten():
            # If node outside of sample area it is ignored
            if node.sample_name == "NaN":
                continue
            # If node has a neighbour to left, and this is not outside of area then we append the
            # difference to the regulariser terms.
            if node.left_neighbour and node.left_neighbour.sample_name != "NaN":
                # TODO: Make difference a log-ratio
                self._regularizer_terms.append(
                    node.concentration - node.left_neighbour.concentration
                )
            # If node has a neighbour above, and this is not outside of area then we append the
            # difference to the regulariser terms.
            if node.top_neighbour and node.top_neighbour.sample_name != "NaN":
                # TODO: Make difference a log-ratio
                self._regularizer_terms.append(
                    node.concentration - node.top_neighbour.concentration
                )

    def _build_regularizer_terms_discrete(self) -> None:
        # Build regularizer
        for _, data in self.sample_network.nodes(data=True):
            concen = data["data"].my_tracer_value
            # Data is divided by the mean as part of .solve method, thus the mean value is simply 1.
            # To calculate (convex) relative differences of observation x from the mean we thus
            # calculate max(x/1,1/x) = max(x,1/x)
            self._regularizer_terms.append(cp.maximum(concen, cp.inv_pos(concen)))

    def _build_problem(self) -> None:
        assert self._primary_terms

        # Build the objective and constraints
        objective = cp.norm(cp.vstack(self._primary_terms))
        if self._regularizer_terms:
            objective += self._regularizer_strength * cp.norm(cp.vstack(self._regularizer_terms))
        constraints = self._constraints

        # Create and solve the problem
        print("Compiling problem...")
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        assert self._problem.is_dcp(dpp=True)

    def _set_observation_parameters(self, observation_data: ElementData) -> None:
        """Resets observation parameters and sets their value according to input observations"""
        obs_mean: float = geo_mean(list(observation_data.values()))
        # Reset all sites' observations
        for x in self._site_to_observation.values():
            x.value = None
        # Assign each observed value to a site, making sure that the site exists
        for site, value in observation_data.items():
            assert site in self._site_to_observation
            # Normalise observation by mean
            self._site_to_observation[site].value = value / obs_mean

        # Ensure that all sites in the problem were assigned
        for x in self._site_to_observation.values():
            assert x.value is not None

    def _set_export_rate_parameters(self, export_rates: Optional[ExportRateData] = None):
        """Resets export rate parameters and sets their value according to input observations"""
        # Reset all sites' export rates
        for x in self._site_to_export_rate.values():
            x.value = None

        # If export_rates provided, assign each one to a site, making sure that the site exists
        if export_rates:
            for site, value in export_rates.items():
                assert site in self._site_to_export_rate
                self._site_to_export_rate[site].value = value
        # Else, export rate is set to default value of 1
        else:
            for x in self._site_to_export_rate.values():
                x.value = 1
        # Ensure that all sites in the problem have a prod rate assigned
        for x in self._site_to_export_rate.values():
            assert x.value is not None

    def _set_inverse_total_flux_parameters(self):
        """Resets inverse total flux dummy parameters and sets their value according to input observations"""
        # Reset all sites' inverse total flux parameters
        for x in self._site_to_inverse_total_flux.values():
            x.value = None

        for site, data in self.sample_network.nodes(data=True):
            assert site in self._site_to_inverse_total_flux
            self._site_to_inverse_total_flux[site].value = (1 / data["data"].my_total_flux).value

        for x in self._site_to_inverse_total_flux.values():
            assert x.value is not None

    def solve(
        self,
        observation_data: ElementData,
        export_rates: Optional[ExportRateData] = None,
        regularization_strength: Optional[float] = None,
        solver: str = "gurobi",
    ) -> Union[Tuple[ElementData, ElementData], Tuple[ElementData, np.ndarray]]:
        self._set_observation_parameters(observation_data=observation_data)
        self._set_export_rate_parameters(export_rates=export_rates)
        self._set_inverse_total_flux_parameters()

        if self._regularizer_terms and not regularization_strength:
            raise Exception("WARNING: Regularizer terms present but no strength assigned.")
        self._regularizer_strength.value = regularization_strength

        # Solvers that can handle this problem type include:
        # ECOS, SCS
        # See: https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
        # See: https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        solvers = {
            # VERY SLOW, probably don't use
            "scip": {
                "solver": cp.SCIP,
                "verbose": True,
            },
            "ecos": {
                "solver": cp.ECOS,
                "verbose": False,
                "max_iters": 10000,
                "abstol_inacc": 5e-5,
                "reltol_inacc": 5e-5,
                "feastol_inacc": 1e-4,
            },
            "scs": {"solver": cp.SCS, "verbose": True, "max_iters": 10000},
            "gurobi": {"solver": cp.GUROBI, "verbose": False, "NumericFocus": 3},
        }
        objective_value = self._problem.solve(**solvers[solver])
        print(
            "{color}Status = {status}\033[39m".format(
                color="" if self._problem.status == "optimal" else "\033[91m",
                status=self._problem.status,
            )
        )
        print(f"Objective value = {objective_value}")

        # Return outputs
        obs_mean: float = geo_mean(list(observation_data.values()))

        downstream_preds = self.get_downstream_prediction_dictionary()
        downstream_preds = {sample: value * obs_mean for sample, value in downstream_preds.items()}
        # If solving continuously return a map on resolution base DEM

        if self.continuous:
            upstream_preds = self.get_upstream_prediction_map() * obs_mean
        # If solving discrete return a dictionary of values corresponding to each sample site
        else:
            upstream_preds = self.get_upstream_prediction_dictionary()
            upstream_preds = {sample: value * obs_mean for sample, value in upstream_preds.items()}

        return downstream_preds, upstream_preds

    def solve_montecarlo(
        self,
        observation_data: ElementData,
        relative_error: float,
        num_repeats: int,
        regularization_strength: Optional[float] = None,
        solver: str = "gurobi",
    ):
        predictions_down_mc = defaultdict(list)

        if self.continuous:
            predictions_up_mc = []
        else:
            predictions_up_mc = defaultdict(list)

        for _ in range(num_repeats):
            observation_data_resampled = {
                sample: value * np.random.normal(loc=1, scale=relative_error / 100)
                for sample, value in observation_data.items()
            }
            element_pred_down, element_pred_upstream = self.solve(
                observation_data=observation_data_resampled,
                solver=solver,
                regularization_strength=regularization_strength,
            )  # Solve problem
            for sample_name in element_pred_down.keys():
                predictions_down_mc[sample_name] += [element_pred_down[sample_name]]

            if self.continuous:
                predictions_up_mc += [element_pred_upstream]
            else:
                for sample_name in element_pred_down.keys():
                    predictions_up_mc[sample_name] += [element_pred_upstream[sample_name]]

        return predictions_down_mc, predictions_up_mc

    def get_downstream_prediction_dictionary(self) -> ElementData:
        # Print the solution we found
        predictions: ElementData = {}
        for sample_name, data in self.sample_network.nodes(data=True):
            data = data["data"]
            predictions[sample_name] = data.my_total_tracer_flux.value / data.my_total_flux.value
        return predictions

    def get_upstream_prediction_dictionary(self) -> ElementData:
        if self.continuous:
            raise Exception(
                "Warning: `get_upstream_prediction_dictionary` only valid for discrete networks"
            )
        # Get the predicted upstream concentration we found
        predictions: ElementData = {}
        for sample_name, data in self.sample_network.nodes(data=True):
            data = data["data"]
            predictions[sample_name] = data.my_tracer_value.value
        return predictions

    def get_upstream_prediction_map(self) -> np.ndarray:
        if not self.continuous:
            raise Exception(
                "Warning: `get_upstream_prediction_map` only valid for continuous networks"
            )
        out = np.zeros(self.grid.area_labels.shape)
        xstep = out.shape[1] / self.grid.nx
        ystep = out.shape[0] / self.grid.ny
        # Loop through inversion grid nodes
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                # indices which subdivide the areas on the base array for each inversion grid.
                x_start = int(i * xstep)
                x_end = int((i + 1) * xstep)
                y_start = int((j * ystep))
                y_end = int((j + 1) * ystep)
                node = self.grid.node_arr[j, i]
                # Catch exception for nodes outside of area
                if node.concentration:
                    val = node.concentration.value
                else:
                    val = np.nan
                out[y_start:y_end, x_start:x_end] = val
        return out

    def get_misfit(self) -> float:
        return cp.norm(cp.vstack(self._primary_terms)).value

    def get_roughness(self) -> float:
        return cp.norm(cp.vstack(self._regularizer_terms)).value


@dataclass
class InverseNode:
    """A single node on an inversion grid.

    Each `InverseNode` corresponds to a rectangle of pixels on the base raster
    and is associated with a single sample site. The sample site which each
    `InverseNode` is associated to is based on sub-catchment which the *centre*
    of the rectangle lies in. For rectangles which overlap two catchments, there
    may be some inaccuracies as each `InverseNode` can only be associated with
    one sample site. As the resolution increases this inaccuracy decreases.

    Attributes:
        left_neighbour : Pointer to left-neighbouring InverseNode
        top_neighbour : Pointer to vertically-above InverseNode
        sample_name : Samplename associated with this node
        concentration: Value to be optimised for
    """

    left_neighbour: "InverseNode"
    top_neighbour: "InverseNode"
    sample_name: str
    concentration: cp.Variable = field(default_factory=lambda: cp.Variable(pos=True))


class InverseGrid:
    """A regularly spaced rectangular grid of inverse nodes

    Args:
        nx : Number of columns in the grid
        ny : Number of rows in the grid
        area_labels : 2D array which matches upstream areas to labels
        sample_network : Network of sample_sites along drainage, with associated data

    Attributes:
        nx : Number of columns in the grid
        ny : Number of rows in the grid
        area_labels : 2D array which matches pixels to sample labels
        node_arr : List of lists (dims: (ny,xs)) containing all InverseNodes
        sites_to_nodes : Dict mapping sample numbers to list of nodes in its upstream area
    """

    def __init__(self, nx: int, ny: int, area_labels: np.array, sample_network: nx.DiGraph) -> None:
        self.area_labels = area_labels
        if nx <= 0 or ny <= 0:
            raise Exception("Warning: nx or ny must be strictly positive")
        xmax = area_labels.shape[1]
        ymax = area_labels.shape[0]
        if ny > ymax or nx > xmax:
            raise Exception(
                "Warning: desired resolution greater than that of DEM. \n Decrease resolution to resolve"
            )
        self.nx = nx
        self.ny = ny
        xstep = xmax / nx
        ystep = ymax / ny
        # The x and y coordinates on the DEM of the *centres* of the rectangular nodes
        xs = np.linspace(start=xstep / 2, stop=xmax - xstep / 2, num=nx)
        ys = np.linspace(start=ystep / 2, stop=ymax - ystep / 2, num=ny)
        self.sites_to_nodes = defaultdict(list)
        # Map area labels to sample numbers
        area_label_to_sample_name = {
            data["data"].label: node for node, data in sample_network.nodes(data=True)
        }
        self.node_arr = np.empty((ny, nx), dtype=object)
        # Loop through a (nx, ny) grid
        for i, x_coord in enumerate(xs):
            for j, y_coord in enumerate(ys):
                # Point towards neighbours, catching uppper & left boundary node exceptions
                left = self.node_arr[j, i - 1] if i > 0 else None
                top = self.node_arr[j - 1, i] if j > 0 else None
                label = self.area_labels[int(y_coord), int(x_coord)]
                sample_name = area_label_to_sample_name[label] if label != 0 else "NaN"
                # Create an inversion node
                node = InverseNode(left_neighbour=left, top_neighbour=top, sample_name=sample_name)
                self.node_arr[j, i] = node
                self.sites_to_nodes[node.sample_name].append(node)
        # For low density grids, sample areas can contain no nodes resulting in errors.
        # Catch this exception here
        num_keys = len(self.sites_to_nodes)
        if num_keys < len(np.unique(self.area_labels)) - 1:
            raise Exception(
                "Warning: Not all catchments contain a node. \n \t Increase resolution to resolve"
            )


def get_element_obs(element: str, obs_data: pd.DataFrame) -> ElementData:
    # TODO(rbarnes): remove the `isinstance`
    element_data: ElementData = {
        e: c
        for e, c in zip(obs_data[SAMPLE_CODE_COL_NAME].tolist(), obs_data[element].tolist())
        if isinstance(c, float)
    }
    return element_data


def mix_downstream(
    sample_network: nx.DiGraph,
    areas: Dict[str, np.ndarray],
    concentration_map: np.ndarray,
    export_rates: Optional[ExportRateData] = None,
) -> Tuple[ElementData, ElementData]:
    """Mixes a given concentration map along drainage, predicting the downstream concentration at sample sites
    Args:
        sample_network: A sample_network of localities (see `get_sample_graphs`)
        areas: A dictionary mapping sample names to sub-basins (see `get_unique_upstream_areas`)
        concentration_map: A 2D map of concentrations which is to be mixed along drainage. Must have same dimensions
        as base flow-direction map/DEM
        export_rates: Dictionary of export rates for each sub-catchment. Defaults to equal export rate in each sub-catchment.
    Returns:
        mixed_downstream_pred: Dictionary containing predicted downstream mixed concentration at each sample sites
        mixed_upstream_pred: Dictionary containing the average concentration of `concentration_map` in each sub-basin
    """
    mixed_downstream_pred: ElementData = {}
    mixed_upstream_pred: ElementData = {}

    for _, data in sample_network.nodes(data=True):
        data["data"].my_total_tracer_flux = 0.0
        data["data"].my_total_flux = 0.0

    for sample_name, my_data in nx_topological_sort_with_data(sample_network):
        # If provided, set export rates from user input
        # else default to equal rate (absolute value is arbitrary)

        my_data.my_export_rate = export_rates[sample_name] if export_rates else 1

        my_data.my_tracer_value = np.mean(concentration_map[areas[sample_name]])
        # area weighted total contribution of material from this node
        my_data.my_flux = my_data.area * my_data.my_export_rate
        # Add the flux I generate to the total flux passing through me
        my_data.my_total_flux += my_data.my_flux

        # area weighted contribution of *tracer* from this node
        my_data.my_tracer_flux = my_data.my_flux * my_data.my_tracer_value
        # Add the *tracer* flux I generate to the total flux of *tracer* passing through me
        my_data.my_total_tracer_flux += my_data.my_tracer_flux

        normalised = my_data.my_total_tracer_flux / my_data.my_total_flux
        mixed_downstream_pred[sample_name] = normalised
        mixed_upstream_pred[sample_name] = my_data.my_tracer_value
        if ds := nx_get_downstream(sample_network, sample_name):
            downstream_data = sample_network.nodes[ds]["data"]
            # Add our flux to downstream node's
            downstream_data.my_total_flux += my_data.my_total_flux
            # Add our *tracer* flux to the downstream node's
            downstream_data.my_total_tracer_flux += my_data.my_total_tracer_flux

    return mixed_downstream_pred, mixed_upstream_pred


def get_unique_upstream_areas(sample_network: nx.DiGraph) -> Dict[str, np.ndarray]:
    """Generates a dictionary which maps sample numbers onto
    the unique upstream area (as a boolean mask)
    for the sample site."""
    I = plt.imread("labels.tif")[:, :, 0]
    return {node: I == data["data"].label for node, data in sample_network.nodes(data=True)}


def plot_sweep_of_regularizer_strength(
    sample_network: nx.DiGraph,
    element_data: ElementData,
    min_: float,
    max_: float,
    trial_num: float,
):
    vals = np.logspace(min_, max_, num=trial_num)  # regularizer strengths to try
    for val in vals:
        print(20 * "_")
        print("Trying regularizer strength: 10^", round(np.log10(val), 3))
        _, _ = sample_network.solve(element_data, solver="ecos", regularization_strength=val)
        roughness = sample_network.get_roughness()
        misfit = sample_network.get_misfit()
        print("Roughness:", np.round(roughness, 4))
        print("Data misfit:", np.round(misfit, 4))
        plt.scatter(roughness, misfit, c="grey")
        plt.text(roughness, misfit, str(round(np.log10(val), 3)))
    plt.xlabel("Roughness")
    plt.ylabel("Data misfit")
    plt.show()


def get_upstream_concentration_map(areas, upstream_preds):
    """Generates a two-dimensional map displaying the predicted upstream
    concentration for a given element for each unique upstream area.
        areas: Dictionary mapping sample numbers onto a boolean mask
               (see `get_unique_upstream_areas`)
        upstream_preds: Dictionary of predicted upstream concentrations
               (see `get_upstream_prediction_dictionary`)
        elem: String of element symbol"""

    out = np.zeros(list(areas.values())[0].shape)  # initialise output
    for sample_name, value in upstream_preds.items():
        out[areas[sample_name]] += value
    return out


def visualise_downstream(pred_dict, obs_dict, element: str) -> None:
    obs = []
    pred = []
    for sample in obs_dict.keys():
        obs += [obs_dict[sample]]
        pred += [pred_dict[sample]]
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    plt.scatter(x=obs, y=pred)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Observed " + element + " concentration mg/kg")
    plt.ylabel("Predicted " + element + " concentration mg/kg")
    plt.plot([0, 1e6], [0, 1e6], alpha=0.5, color="grey")
    plt.xlim((np.amin(obs * 0.9), np.amax(obs * 1.1)))
    plt.ylim((np.amin(pred * 0.9), np.amax(pred * 1.1)))
    ax = plt.gca()
    ax.set_aspect(1)


def process_data(
    flowdirs_filename: str,
    data_filename: str,
    excluded_elements: Optional[List[str]] = None,
) -> pd.DataFrame:
    sample_network, sample_adjacency = get_sample_graphs(flowdirs_filename, data_filename)

    plot_network(sample_network)
    obs_data = pd.read_csv(data_filename, delimiter=" ")
    obs_data = obs_data.drop(columns=excluded_elements)

    problem = SampleNetwork(sample_network=sample_network, sample_adjacency=sample_adjacency)

    get_unique_upstream_areas(problem.sample_network)

    results = None
    # TODO(r-barnes,alexlipp): Loop over all elements once we achieve acceptable results
    for element in ELEMENT_LIST[0:20]:
        if element not in obs_data.columns:
            continue

        print(f"\n\033[94mProcessing element '{element}'...\033[39m")

        element_data = get_element_obs(element=element, obs_data=obs_data)
        try:
            predictions, _ = problem.solve(
                element_data, solver="ecos", regularization_strength=1e-3
            )
        except cp.error.SolverError as err:
            print(f"\033[91mSolver Error - skipping this element!\n{err}")
            continue

        if results is None:
            results = pd.DataFrame(element_data.keys())
        results[element + "_obs"] = [element_data[sample] for sample in element_data.keys()]
        results[element + "_dwnst_prd"] = [predictions[sample] for sample in element_data.keys()]

    return results


def main():
    results = process_data(
        flowdirs_filename="data/d8.asc",
        data_filename="data/sample_data.dat",
        excluded_elements=["Bi", "S"],
    )
    print(results)


if __name__ == "__main__":
    main()
