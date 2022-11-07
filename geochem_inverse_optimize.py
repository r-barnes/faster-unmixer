#!/usr/bin/env python3

import os
import tempfile
from typing import Dict, Final, Iterator, List, Optional, Tuple

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


def cp_log_ratio_norm(a, b):
    return cp.maximum(a / b, b * cp.inv_pos(a))


def nx_topological_sort_with_data(
    G: nx.DiGraph,
) -> Iterator[Tuple[str, pyfastunmix.SampleNode]]:
    return ((x, G.nodes[x]["data"]) for x in nx.topological_sort(G))


def nx_get_downstream(G: nx.DiGraph, x: str) -> str:
    """Gets the downstream child from a node with only one child"""
    s = list(G.successors(x))
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
    data_dir: str,
) -> Tuple[nx.DiGraph, "pyfastunmix.SampleAdjacency"]:
    # Get the graph representations of the data
    sample_network_raw, sample_adjacency = pyfastunmix.fastunmix(data_dir)

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
    # TODO: Docstrings
    def __init__(
        self,
        sample_network: nx.DiGraph,
        sample_adjacency: "pyfastunmix.SampleAdjacency",
    ) -> None:
        self.sample_network = sample_network
        self.sample_adjacency = sample_adjacency
        self._site_to_parameter: Dict[str, cp.Parameter] = {}
        self._primary_terms = []
        self._regularizer_terms = []
        self._regularizer_strength = cp.Parameter(nonneg=True)
        self._problem = None
        self._build_primary_terms()
        self._build_regularizer_terms()
        self._build_problem()

    def _build_primary_terms(self) -> None:
        for _, data in self.sample_network.nodes(data=True):
            data["data"].total_flux = 0.0

        # Build the main objective
        # Use a topological sort to ensure an upstream-to-downstream traversal
        for sample_name, my_data in nx_topological_sort_with_data(self.sample_network):
            print(sample_name)
            # Set up a CVXPY parameter for each element for each node
            my_data.my_value = cp.Variable(pos=True)

            # area weighted contribution from this node
            my_data.my_flux = my_data.area * my_data.my_value

            # Add the flux I generate to the total flux passing through me
            my_data.total_flux += my_data.my_flux

            observed = cp.Parameter(pos=True)
            self._site_to_parameter[my_data.name] = observed
            normalised_concentration = my_data.total_flux / my_data.total_upstream_area
            self._primary_terms.append(cp_log_ratio_norm(normalised_concentration, observed))

            if ds := nx_get_downstream(self.sample_network, sample_name):
                downstream_data = self.sample_network.nodes[ds]["data"]
                # Add our flux to the downstream node's
                downstream_data.total_flux += my_data.total_flux

    def _build_regularizer_terms(self) -> None:
        # Build the regularizer
        for adjacent_nodes, border_length in self.sample_adjacency.items():
            a_concen = self.sample_network.nodes[adjacent_nodes[0]]["data"].my_value
            b_concen = self.sample_network.nodes[adjacent_nodes[1]]["data"].my_value
            # TODO: Make difference a log-ratio
            # self._regularizer_terms.append(border_length * (cp_log_ratio_norm(a_concen,b_concen)))
            # Simple difference (not desirable)
            self._regularizer_terms.append(border_length * (a_concen - b_concen))

    def _build_problem(self) -> None:
        assert self._primary_terms

        if not self._regularizer_terms:
            print("WARNING: No regularizer terms found!")

        # Build the objective and constraints
        objective = cp.norm(cp.vstack(self._primary_terms))
        if self._regularizer_terms:
            objective += self._regularizer_strength * cp.norm(cp.vstack(self._regularizer_terms))
        constraints = []

        # Create and solve the problem
        print("Compiling problem...")
        self._problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve(
        self,
        observation_data: ElementData,
        regularization_strength: Optional[float] = None,
        solver: str = "gurobi",
    ):
        obs_mean = np.mean(list(observation_data.values()))

        # Reset all sites' observations
        for x in self._site_to_parameter.values():
            x.value = None
        # Assign each observed value to a site, making sure that the site exists
        for site, value in observation_data.items():
            assert site in self._site_to_parameter
            # Normalise observation by mean
            self._site_to_parameter[site].value = value / obs_mean
        # Ensure that all sites in the problem were assigned
        for x in self._site_to_parameter.values():
            assert x.value is not None

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
                "verbose": True,
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
        downstream_preds = get_downstream_prediction_dictionary(sample_network=self.sample_network)
        downstream_preds.update(
            (sample, value * obs_mean) for sample, value in downstream_preds.items()
        )

        upstream_preds = get_upstream_prediction_dictionary(sample_network=self.sample_network)
        upstream_preds.update(
            (sample, value * obs_mean) for sample, value in downstream_preds.items()
        )

        return downstream_preds, upstream_preds


class SampleNetworkContinuous:
    # TODO: Docstrings
    def __init__(self, sample_network: nx.DiGraph, area_labels: np.array, nx: int, ny: int) -> None:
        self.sample_network = sample_network
        self.grid = InverseGrid(nx, ny, area_labels)
        self._site_to_parameter: Dict[str, cp.Parameter] = {}
        self._primary_terms = []
        self._regularizer_terms = self.grid.get_regularizer_terms()
        self._regularizer_strength = cp.Parameter(nonneg=True)
        self._problem = None
        self._build_primary_terms()
        self._build_problem()

    def _build_primary_terms(self) -> None:
        for _, data in self.sample_network.nodes(data=True):
            data["data"].total_flux = 0.0

        # Build the main objective
        # Use a topological sort to ensure an upstream-to-downstream traversal

        # TODO: Delete this legend when labels.tif is updated
        # Set up a number <-> string legend for upstream areas
        sample_num_dict = {}
        i = 1
        for sample_name in nx.topological_sort(self.sample_network):
            sample_num_dict[sample_name] = i
            i += 1
        for sample_name, my_data in nx_topological_sort_with_data(self.sample_network):
            # # TODO: Update this to work with new labels.tif output
            samp_num = sample_num_dict[sample_name]
            concs = [
                node.concentration for node in self.grid.node_sample_dict[samp_num]
            ]  # TODO: Update above to work with new labels.tif output

            my_data.my_value = cp.sum(concs) / len(
                concs
            )  # mean conc of all inversion nodes upstream

            # area weighted contribution from this node
            my_data.my_flux = my_data.area * my_data.my_value

            # Add the flux I generate to the total flux passing through me
            my_data.total_flux += my_data.my_flux

            observed = cp.Parameter(pos=True)
            self._site_to_parameter[my_data.name] = observed
            normalised_concentration = my_data.total_flux / my_data.total_upstream_area
            self._primary_terms.append(cp_log_ratio_norm(normalised_concentration, observed))

            if ds := nx_get_downstream(self.sample_network, sample_name):
                downstream_data = self.sample_network.nodes[ds]["data"]
                # Add our flux to the downstream node's
                downstream_data.total_flux += my_data.total_flux

    def _build_problem(self) -> None:
        assert self._primary_terms
        if not self._regularizer_terms:
            print("WARNING: No regularizer terms found!")

        # Build the objective and constraints
        objective = cp.norm(cp.vstack(self._primary_terms))
        if self._regularizer_terms:
            objective += self._regularizer_strength * cp.norm(cp.vstack(self._regularizer_terms))
        constraints = []

        # Create and solve the problem
        print("Compiling problem...")
        self._problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve(
        self,
        observation_data: ElementData,
        regularization_strength: Optional[float] = None,
        solver: str = "gurobi",
    ):
        obs_mean = np.mean(list(observation_data.values()))

        # Reset all sites' observations
        for x in self._site_to_parameter.values():
            x.value = None
        # Assign each observed value to a site, making sure that the site exists
        for site, value in observation_data.items():
            assert site in self._site_to_parameter
            # Normalise observation by mean
            self._site_to_parameter[site].value = value / obs_mean
        # Ensure that all sites in the problem were assigned
        for x in self._site_to_parameter.values():
            assert x.value is not None

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
                "verbose": True,
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
        downstream_preds = get_downstream_prediction_dictionary(sample_network=self.sample_network)
        downstream_preds.update(
            (sample, value * obs_mean) for sample, value in downstream_preds.items()
        )

        upstream_preds = self.grid.get_upstream_map() * obs_mean
        return downstream_preds, upstream_preds


class InverseNode:
    """A single node on an inversion grid.

    Args:
        x (float): x-coordinate for node
        y (float): y-coordinate for node

    Attributes:
        x (float): x-coordinate for node
        y (float): y-coordinate for node
        left_neighbour (InverseNode): Pointer to left-neighbouring InverseNode
        top_neighbour (InverseNode): Pointer to vertically-above InverseNode
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.concentration = None
        self.left_neighbour = None
        self.top_neighbour = None
        self.sample_num = None


class InverseGrid:
    """A regularly spaced rectangular grid of inverse nodes

    Args:
        nx (int) : Number of columns in the grid
        ny (int) : Number of rows in the grid
        area_labels (np.array) : 2D array which matches upstream areas to sample numbers

    Attributes:
        nx (int) : Number of columns in the grid
        ny (int) : Number of rows in the grid
        area_labels (np.array): 2D array which matches upstream areas to sample numbers
        xs (np.array) : 1D array of x-coordinates for inverse nodes
        ys (np.array) : 1D array of y-coordinates for inverse nodes
        node_coord_dict (tuple : InverseNode): Dict mapping coordinates tuple to corresponding InverseNode
        node_sample_dict (float : list of InverseNode): Dict mapping sample numbers to list of nodes in its upstream area
    """

    def __init__(self, nx: int, ny: int, area_labels: np.array) -> None:
        self.area_labels = area_labels
        self.nx = nx
        self.ny = ny
        xmax = area_labels.shape[1]
        ymax = area_labels.shape[0]
        xstep = xmax / nx
        ystep = ymax / ny
        self.xs = np.linspace(start=xstep / 2, stop=xmax - xstep / 2, num=nx)
        self.ys = np.linspace(start=ystep / 2, stop=ymax - ystep / 2, num=ny)
        self.node_coord_dict = {}
        self.node_sample_dict = {}
        # Loop through a (nx, ny) grid
        for i in range(self.nx):
            for j in range(self.ny):
                x_coord = self.xs[i]
                y_coord = self.ys[j]
                # Create an inversion node
                node = InverseNode(x=x_coord, y=y_coord)
                self.node_coord_dict[(x_coord, y_coord)] = node
                # Catch exception of left most nodes (used for roughness calculation)
                if not i == 0:
                    # Point towards neighbour
                    node.left_neighbour = self.node_coord_dict[(self.xs[i - 1], y_coord)]
                # Catch exception of upper most nodes
                if not j == 0:
                    # Point towards neighbour (used for roughness calculation)
                    node.top_neighbour = self.node_coord_dict[(x_coord, self.ys[j - 1])]
                # Set the sample number based off area map
                node.sample_num = self.area_labels[int(np.floor(node.y)), int(np.floor(node.x))]
                # Only assign variables for nodes within the sampled area
                if not node.sample_num == 0:
                    node.concentration = cp.Variable(pos=True)
                    # If sample already has upstream areas associated with it
                    if node.sample_num in self.node_sample_dict.keys():
                        self.node_sample_dict[node.sample_num].append(node)
                    # Initiate the node list for that sample
                    else:
                        self.node_sample_dict[node.sample_num] = []
                        self.node_sample_dict[node.sample_num].append(node)
        # For low density grids, sample areas can contain no nodes resulting in errors.
        # Catch this exception here
        num_keys = len(self.node_sample_dict.keys())
        if num_keys < (len(np.unique(self.area_labels)) - 1):
            raise Exception(
                "Warning: Not all catchments contain a node. \n \t Increase resolution to resolve"
            )

    def get_regularizer_terms(self) -> None:
        regularizer_terms = []
        # Loop through all nodes in grid
        for _, node in self.node_coord_dict.items():
            # If node outside of sample area it is ignored
            if node.sample_num == 0:
                continue
            # If node has a neighbour to left, and this is not outside of area then we append the
            # difference to the regulariser terms.
            if node.left_neighbour and not (node.left_neighbour.sample_num == 0):
                # TODO: Make difference a log-ratio
                regularizer_terms.append(node.concentration - node.left_neighbour.concentration)
            # If node has a neighbour above, and this is not outside of area then we append the
            # difference to the regulariser terms.
            if node.top_neighbour and not (node.top_neighbour.sample_num == 0):
                # TODO: Make difference a log-ratio
                regularizer_terms.append(node.concentration - node.top_neighbour.concentration)
        return regularizer_terms

    def get_upstream_map(self) -> np.array:
        out = np.zeros(self.area_labels.shape)
        xstep = out.shape[1] / self.nx
        ystep = out.shape[0] / self.ny
        out = np.zeros(self.area_labels.shape)
        # Loop through inversion grid nodes
        for i in np.arange(self.nx):
            for j in np.arange(self.ny):
                # indices which subdivide the areas on the base array for each inversion grid.
                x_start = int(np.floor(i * xstep))
                x_end = int(np.floor((i + 1) * xstep))
                y_start = int(np.floor(j * ystep))
                y_end = int(np.floor((j + 1) * ystep))
                node = self.node_coord_dict[(self.xs[i], self.ys[j])]
                # Catch exception for nodes outside of area
                if node.concentration:
                    val = node.concentration.value
                else:
                    val = np.nan
                out[y_start:y_end, x_start:x_end] = val
        return out


def get_downstream_prediction_dictionary(sample_network: nx.DiGraph) -> pd.DataFrame:
    # Print the solution we found
    predictions: ElementData = {}
    for sample_name, data in sample_network.nodes(data=True):
        data = data["data"]
        predictions[sample_name] = data.total_flux.value / data.total_upstream_area

    return predictions


def get_upstream_prediction_dictionary(sample_network: nx.DiGraph) -> pd.DataFrame:
    # Get the predicted upstream concentration we found
    predictions: ElementData = {}
    for sample_name, data in sample_network.nodes(data=True):
        data = data["data"]
        predictions[sample_name] = data.my_value.value
    return predictions


def get_element_obs(element: str, obs_data: pd.DataFrame) -> ElementData:
    # TODO(rbarnes): remove the `isinstance`
    element_data: ElementData = {
        e: c
        for e, c in zip(obs_data[SAMPLE_CODE_COL_NAME].tolist(), obs_data[element].tolist())
        if isinstance(c, float)
    }
    return element_data


def get_unique_upstream_areas(sample_network: nx.DiGraph):
    """Generates a dictionary which maps sample numbers onto
    the unique upstream area (as a boolean mask)
    for the sample site."""
    I = plt.imread("labels.tif")[:, :, 0]
    areas = {}
    counter = 1
    for node in sample_network.nodes:
        areas[node] = I == counter
        counter += 1
    return areas


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
    data_dir: str, data_filename: str, excluded_elements: Optional[List[str]] = None
) -> pd.DataFrame:
    sample_network, sample_adjacency = get_sample_graphs(data_dir)

    plot_network(sample_network)
    obs_data = pd.read_csv(data_filename, delimiter=" ")
    obs_data = obs_data.drop(columns=excluded_elements)

    problem = SampleNetwork(sample_network=sample_network, sample_adjacency=sample_adjacency)

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
        data_dir="data/",
        data_filename="data/geochem_no_dupes.dat",
        excluded_elements=["Bi", "S"],
    )
    print(results)


if __name__ == "__main__":
    main()
