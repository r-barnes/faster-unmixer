#!/usr/bin/env python3
import logging
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    DefaultDict,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# TODO(rbarnes): Make a requirements file for conda
import cvxpy as cp

# pyre-fixme[21]: Could not find module `matplotlib.image`.
import matplotlib.image as mpimg

# pyre-fixme[21]: Could not find module `matplotlib.pyplot`.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm

import _funmixer_native as fn

from .cvxpy_extensions import ReciprocalParameter, cp_log_ratio

NO_DOWNSTREAM: Final[int] = 0
SAMPLE_CODE_COL_NAME: Final[str] = "Sample.Code"
ELEMENT_LIST: Final[List[str]] = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]  # fmt: skip

ElementData = Dict[str, float]
ExportRateData = Dict[str, float]

T = TypeVar("T")


def not_none(x: Optional[T]) -> T:
    assert x is not None
    return x


@dataclass
class SampleNode:
    """
    A class to represent a sample node in a Sample Network.
    """

    name: str
    x: int
    y: int
    downstream_node: Optional[str]
    upstream_nodes: List[str]
    area: float
    total_upstream_area: int
    # Properties added dynamically by Python
    label: int
    my_flux: Optional[Union[float, cp.Expression]] = None
    my_total_flux: Union[float, cp.Expression] = 0.0
    my_total_tracer_flux: Union[float, cp.Expression] = 0.0
    my_tracer_flux: Optional[cp.Expression] = None
    my_tracer_value: Optional[cp.Expression] = None
    rltv_area: float = -np.inf  # Obviously bad value
    total_flux: Optional[cp.Expression] = None
    my_export_rate: cp.Parameter = field(default_factory=lambda: cp.Parameter(pos=True))

    @classmethod
    def from_native(cls, n: fn.NativeSampleNode) -> "SampleNode":
        return cls(
            name=n.name,
            x=n.x,
            y=n.y,
            downstream_node=n.downstream_node,
            upstream_nodes=n.upstream_nodes,
            area=n.area,
            total_upstream_area=n.total_upstream_area,
            label=n.label,
        )


@dataclass(frozen=True)
class DownstreamNode:
    node: str
    data: SampleNode


def nx_items(sample_network: nx.DiGraph) -> Iterator[Tuple[str, SampleNode]]:
    for node, data in sample_network.nodes(data=True):
        yield node, data["data"]


def nx_values(sample_network: nx.DiGraph) -> Iterator[SampleNode]:
    for data in sample_network.nodes.values():
        yield data["data"]


def native_sample_graph_to_python(g: Dict[str, fn.NativeSampleNode]) -> Dict[str, SampleNode]:
    """
    Convert a graph of native SampleNodes to a graph of Python SampleNodes.
    """
    newg: Dict[str, SampleNode] = {}
    for k, v in g.items():
        newg[k] = SampleNode.from_native(v)
    return newg


# Solvers that can handle this problem type include:
# ECOS, SCS
# See: https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
# See: https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
SOLVERS: Dict[str, Any] = {
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
    "scs": {"solver": cp.SCS, "verbose": False, "max_iters": 10000},
    "gurobi": {"solver": cp.GUROBI, "verbose": False, "NumericFocus": 3},
}

logger: logging.Logger = logging.getLogger()


@dataclass
class FunmixerSolution:
    """
    A class to hold the results of a SampleNetworkUnmixer run.
    """

    objective_value: float
    solver_name: str
    total_time: float
    solve_time: Optional[float]
    setup_time: Optional[float]
    downstream_preds: ElementData
    upstream_preds: ElementData


def geo_mean(x: List[float]) -> float:
    """
    Returns the geometric mean of a list of numbers.

    Args:
        x: The list of numbers.

    Returns:
        The geometric mean of the numbers in the list.
    """
    # pyre-fixme[6]: For 1st argument expected `Union[bytes, complex, float, int,
    #  generic, str]` but got `List[float]`.
    return np.exp(np.log(x).mean())


def nx_topological_sort_with_data(
    G: nx.DiGraph,
) -> Iterator[Tuple[str, SampleNode]]:
    """
    Returns a topological sort of the graph, with the data of each node.

    Args:
        G: The graph.

    Returns:
        An iterator yielding tuples of node name and node data.
    """
    return (
        (x, cast(SampleNode, G.nodes[x]["data"]))
        for x in cast(Iterator[str], nx.topological_sort(G))
    )


def nx_get_downstream_node(G: nx.DiGraph, x: str) -> Optional[str]:
    """
    Gets the name of the downstream child, if there is one.

    Args:
        G: The graph.
        x: The node.

    Returns:
        The downstream child node's name, or None if there is no downstream child.

    Raises:
        Exception: If there is more than one downstream neighbor.
    """
    s: List[str] = list(G.successors(x))
    if len(s) == 0:
        return None
    elif len(s) == 1:
        return s[0]
    else:
        raise Exception("More than one downstream neighbour!")


def nx_get_downstream_data(G: nx.DiGraph, x: str) -> Optional[SampleNode]:
    """
    Gets data from the downstream child, if there is one

    Args:
        G: The graph.
        x: The node.

    Returns:
        The downstream child's data, or None if there is no child.

    Raises:
        Exception: If there is more than one downstream neighbor.
    """
    s = nx_get_downstream_node(G, x)
    if s:
        return cast(SampleNode, G.nodes[s]["data"])


def plot_network(G: nx.DiGraph) -> None:
    """
    Plots a networkx graph using graphviz.

    Args:
        G: The graph to plot.
    """
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
) -> Tuple[nx.DiGraph, "fn.SampleAdjacency"]:
    """
    Get sample network graph and adjacency matrix from flow direction and concentration dataset files.

    Args:
        flowdirs_filename: File name of the flow directions D8 raster.
        sample_data_filename: File name of the geochemical sample data (concentrations).

    Returns:
        A tuple containing two objects:
        - sample_network: A networkx DiGraph representing the sample network.
        - sample_adjacency: An instance of fn.SampleAdjacency. This contains the length of shared catchment
         between each node's subbasin.

    """
    sample_network_native_raw, sample_adjacency = fn.fastunmix(
        flowdirs_filename, sample_data_filename
    )
    sample_network_raw = native_sample_graph_to_python(sample_network_native_raw)

    # Convert it into a networkx graph for easy use in Python
    sample_network = nx.DiGraph()
    for x in sample_network_raw.values():  # Skip the first node into which it all flows
        if x.name == fn.root_node_name:
            continue
        sample_network.add_node(x.name, data=x)
        if x.downstream_node != fn.root_node_name:
            sample_network.add_edge(x.name, x.downstream_node)

    return sample_network, sample_adjacency


class SampleNetworkUnmixer:
    """
    This class provides functionality to `unmix' a network of samples of concentration data
    to recover the upstream source concentrations.

    Attributes:
        sample_network (nx.DiGraph): The sample network.
        use_regularization (bool): Flag indicating whether to use regularization to solve.

    Methods:
        __init__:
            Initialize the SampleNetworkUnmixer class.
        solve:
            Solve the optimization problem.
        solve_montecarlo:
            Solve the optimization problem with Monte Carlo simulation.
        get_downstream_prediction_dictionary:
            Get the downstream prediction as a dictionary.
        get_upstream_prediction_dictionary:
            Get the upstream prediction as a dictionary.
    """

    def __init__(
        self,
        sample_network: nx.DiGraph,
        use_regularization: bool = True,
    ) -> None:
        """
        Initialize the SampleNetworkUnmixer class.

        Args:
            sample_network (nx.DiGraph): The sample network.
            use_regularization (bool): Flag indicating whether to use regularization.
        """

        self.sample_network = sample_network
        self._site_to_observation: Dict[str, ReciprocalParameter] = {}
        self._site_to_export_rate: Dict[str, cp.Parameter] = {}
        self._site_to_total_flux: Dict[str, ReciprocalParameter] = {}
        self._primary_terms: List[cp.Expression] = []
        self._regularizer_terms: List[cp.Expression] = []
        self._constraints: List[cp.Constraint] = []
        self._regularizer_strength = cp.Parameter(nonneg=True)
        self._problem: Optional[cp.Problem] = None
        self._build_primary_terms()
        if use_regularization:
            self._build_regularizer_terms()
        self._build_problem()

    def _calculate_normalised_areas(self, sample_network: nx.DiGraph) -> None:
        """
        Adds a new attribute `rltv_area` to each node, representing the upstream area of the node divided by the mean upstream area
        of all nodes in the network.

        Args:
            sample_network: The sample network graph.

        Note:
            The method calculates the mean upstream area of all nodes in the network and assigns a normalized relative area (`rltv_area`)
            to each node in the graph based on its individual upstream area divided by the mean area. This step improves numerical accuracy
            and does not affect the results as all values are divided by a constant.
        """
        areas = [node.area for node in nx_values(sample_network)]
        mean_area = float(np.mean(np.array(areas)))

        for node in nx_values(sample_network):
            node.rltv_area = node.area / mean_area

    def _build_primary_terms(self) -> None:
        """
        Build the primary terms for the objective function.
        """
        for data in nx_values(self.sample_network):
            data.my_total_flux = 0.0
            data.my_total_tracer_flux = 0.0

        # Normalises node area by total mean to improve numerical accuracy
        self._calculate_normalised_areas(sample_network=self.sample_network)

        # Build the main objective
        # Use a topological sort to ensure an upstream-to-downstream traversal
        for sample_name, my_data in nx_topological_sort_with_data(self.sample_network):
            # Set up a CVXPY parameter for each element for each node
            my_data.my_tracer_value = cp.Variable(pos=True)

            # Export rate of total material (e.g., erosion rate, run-off)
            # Value is set at runtime
            my_data.my_export_rate = cp.Parameter(pos=True)
            self._site_to_export_rate[my_data.name] = my_data.my_export_rate

            # Area weighted total contribution of material from this node
            my_data.my_flux = my_data.my_export_rate * my_data.rltv_area

            # Add the flux I generate to the total flux passing through me
            my_data.my_total_flux += my_data.my_flux
            # Set up a ReciprocalParameter for total flux to make problem DPP.
            # Value of this parameter is set at solve time as it
            # depends on export rate parameter values
            total_flux_dummy = ReciprocalParameter(pos=True)
            self._site_to_total_flux[my_data.name] = total_flux_dummy

            # Area weighted contribution of *tracer* from this node
            # pyre-fixme[58]: `*` is not supported for operand types `Union[None,
            #  cp.expressions.expression.Expression, float]` and
            #  `Optional[cp.expressions.expression.Expression]`.
            my_data.my_tracer_flux = my_data.my_flux * my_data.my_tracer_value
            # Add the *tracer* flux I generate to the total flux of *tracer* passing through me
            my_data.my_total_tracer_flux += my_data.my_tracer_flux

            # Set up a dummy (parameter free) variable that encodes the total *tracer* flux at the node.
            # This ensures that the problem is DPP.
            total_tracer_flux_dummy = cp.Variable(pos=True)
            # We add a constraint that this must equal the parameter encoded `total_tracer_flux`
            self._constraints.append(total_tracer_flux_dummy == my_data.my_total_tracer_flux)

            # Set up a dummy (parameter free) variable for normalised concentration.
            # `.rp` ensures that the problem is DPP.
            normalised_concentration = total_tracer_flux_dummy * total_flux_dummy.rp
            normalised_concentration_dummy = cp.Variable(pos=True)
            # We add a constraint that this must equal the parameter encoded `normalised_concentration`
            self._constraints.append(normalised_concentration_dummy == normalised_concentration)

            # Set up a parameter for the observation at node
            # Value is set at solve time
            observed = ReciprocalParameter(pos=True)
            self._site_to_observation[my_data.name] = observed

            # Calculate misfit and append to primary terms in objective function
            misfit = cp_log_ratio(normalised_concentration_dummy, observed)
            self._primary_terms.append(misfit)

            if (ds := nx_get_downstream_data(self.sample_network, sample_name)) is not None:
                # Add our flux to downstream node's
                ds.my_total_flux += my_data.my_total_flux
                # Add our *tracer* flux to the downstream node's
                ds.my_total_tracer_flux += my_data.my_total_tracer_flux

    def _build_regularizer_terms(self) -> None:
        """
        Build the regularizer terms.
        """
        # Build regularizer
        for data in nx_values(self.sample_network):
            concen = data.my_tracer_value
            # Data is divided by the mean as part of .solve method, thus the mean value is simply 1.
            # To calculate (convex) relative differences of observation x from the mean we thus
            # calculate max(x/1,1/x) = max(x,1/x)
            self._regularizer_terms.append(cp.maximum(concen, cp.inv_pos(concen)))

    def _build_problem(self) -> None:
        """
        Build the optimization problem.
        """
        assert self._primary_terms

        # Build the objective and constraints
        objective = cp.norm(cp.vstack(self._primary_terms))
        if self._regularizer_terms:
            objective += self._regularizer_strength * cp.norm(cp.vstack(self._regularizer_terms))
        constraints = self._constraints

        # Create and solve the problem
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        # pyre-fixme[28]: Unexpected keyword argument `dpp`.
        assert self._problem.is_dcp(dpp=True)

    def _set_observation_parameters(self, observation_data: ElementData) -> None:
        """
        Reset and set the observation parameters according to input observations.

        Args:
            observation_data: The observation data.
        """
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

    def _set_export_rate_parameters(self, export_rates: Optional[ExportRateData] = None) -> None:
        """
        Reset and set the export rate parameters according to input export rates.

        Args:
            export_rate_data: The export rate data.
        """
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

    def _set_total_flux_parameters(self) -> None:
        """
        Reset and set the total flux parameters according to total fluxes calculated in network.
        """
        # Reset all sites' total flux parameters
        for x in self._site_to_total_flux.values():
            x.value = None

        for site, data in nx_items(self.sample_network):
            # pyre-fixme[16]: Item `float` of `Union[Expression, float]` has no
            #  attribute `value`.
            self._site_to_total_flux[site].value = data.my_total_flux.value

        for x in self._site_to_total_flux.values():
            assert x.value is not None

    def solve(
        self,
        observation_data: ElementData,
        export_rates: Optional[ExportRateData] = None,
        regularization_strength: Optional[float] = None,
        solver: str = "ecos",
    ) -> FunmixerSolution:
        """
        Solves the optimization problem.

        This method solves the optimization problem to estimate downstream and upstream predictions
        based on the provided observation data and export rates. The optimization problem is solved
        using the specified solver.

        Args:
            observation_data: The observed data for each element.
            export_rates: The export rates for each element. If not provided these are all set to 1.
            regularization_strength: The strength of the regularization term
            solver: The solver to use for solving the optimization problem (default is ecos)

        Returns:
            A tuple containing the downstream and upstream predictions. The downstream and upstream predictions
            are returned as `ElementData`, which is a dictionary-like object containing the concentrations for each element.

        Raises:
            Exception: If regularizer terms are present but no regularization strength is assigned.

        Notes:
            - The observation data and export rates should be provided as dictionaries or dictionary-like objects,
              where the keys represent the elements and the values represent the corresponding concentration.
            - The regularization strength is used to balance the fit to the observed data and the regularization terms.
              A higher value results in a smoother solution with more emphasis on the regularization terms.
              A lower value results in a solution that fits the observed data more closely but may `overfit' data.
        """

        self._set_observation_parameters(observation_data=observation_data)
        self._set_export_rate_parameters(export_rates=export_rates)
        self._set_total_flux_parameters()

        if self._regularizer_terms and not regularization_strength:
            raise Exception("WARNING: Regularizer terms present but no strength assigned.")
        self._regularizer_strength.value = regularization_strength

        if solver not in SOLVERS:
            raise Exception(
                f"Solver {solver} not supported. Supported solvers are {list(SOLVERS.keys())}"
            )

        assert (problem := self._problem) is not None
        logger.info("Solving problem...")
        start_solve_time = time.time()
        objective_value = problem.solve(**SOLVERS[solver])
        end_solve_time = time.time()

        if problem.status == "optimal":
            logger.info(f"Status = {problem.status}")
        else:
            logger.warning(f"Status = {problem.status}")

        # Return outputs
        obs_mean: float = geo_mean(list(observation_data.values()))

        downstream_preds = self.get_downstream_prediction_dictionary()
        downstream_preds = {sample: value * obs_mean for sample, value in downstream_preds.items()}
        upstream_preds = self.get_upstream_prediction_dictionary()
        upstream_preds = {sample: value * obs_mean for sample, value in upstream_preds.items()}

        logger.info(f"Objective value = {objective_value}")
        logger.info(f"Total time = {end_solve_time - start_solve_time}")
        logger.info(f"Solver name = {problem.solver_stats.solver_name}")
        logger.info(f"Solve time = {problem.solver_stats.solve_time}")
        logger.info(f"Setup time = {problem.solver_stats.setup_time}")

        return FunmixerSolution(
            objective_value=float(objective_value),
            solver_name=problem.solver_stats.solver_name,
            setup_time=problem.solver_stats.setup_time,
            solve_time=problem.solver_stats.solve_time,
            total_time=end_solve_time - start_solve_time,
            upstream_preds=upstream_preds,
            downstream_preds=downstream_preds,
        )

    def solve_montecarlo(
        self,
        observation_data: ElementData,
        relative_error: float,
        num_repeats: int,
        regularization_strength: Optional[float] = None,
        solver: str = "gurobi",
    ) -> Tuple[DefaultDict[str, List[float]], Dict[str, List[float]]]:
        """
        Solves the optimization problem using Monte Carlo simulation.

        This method solves the optimization problem by repeatedly sampling the observation data with random errors
        and solving the problem for each sampled data. Monte Carlo simulation is used to estimate the uncertainty
        in the downstream and upstream predictions.

        Args:
            observation_data: The observed data for each element.
            relative_error: The *relative* error as a percentage to use for resampling the observation data.
            num_repeats: The number of times to repeat the Monte Carlo simulation.
            regularization_strength: The strength of the regularization term (default: None).
            solver: The solver to use for solving the optimization problem (default: "gurobi").

        Returns:
                A tuple containing the Monte Carlo simulation results.
                - Both the downstream and upstream predictions are returned as dictionaries.
                `predictions_down_mc` represents the downstream predictions, and `predictions_up_mc` represents
                the upstream predictions, where each key represents a sample name, and the corresponding value is a list
                of predictions for that sample across the Monte Carlo simulation.

        Notes:
            - The observation data should be provided as a dictionary or a dictionary-like object,
            where the keys represent the elements and the values represent the corresponding observed data.
            - The `relative_error` parameter determines the amount of random error introduced during resampling.
            It should be specified as a percentage value.
            - The `num_repeats` parameter determines the number of Monte Carlo simulation iterations to perform.
            - The regularization strength is used to balance the fit to the observed data and the regularization terms.
            A higher value results in a smoother solution with more emphasis on the regularization terms.
            A lower value results in a solution that fits the observed data more closely but may `overfit' data.
        """
        predictions_down_mc: DefaultDict[str, List[float]] = defaultdict(list)
        predictions_up_mc: DefaultDict[str, List[float]] = defaultdict(list)

        for _ in tqdm.tqdm(range(num_repeats), total=num_repeats):
            observation_data_resampled = {
                sample: value * np.random.normal(loc=1, scale=relative_error / 100)
                for sample, value in observation_data.items()
            }
            solution = self.solve(
                observation_data=observation_data_resampled,
                solver=solver,
                regularization_strength=regularization_strength,
            )  # Solve problem
            for sample_name, v in solution.downstream_preds.items():
                predictions_down_mc[sample_name].append(v)

            for sample_name, v in solution.upstream_preds.items():
                predictions_up_mc[sample_name].append(v)

        return predictions_down_mc, predictions_up_mc

    def get_downstream_prediction_dictionary(self) -> ElementData:
        """
        Returns the downstream predictions as a dictionary.

        This method returns a dictionary containing the downstream predictions for each sample site in the network.
        The keys in the dictionary represent the sample names, and the corresponding values represent the downstream
        predictions for each sample site.

        Returns:
            ElementData: A dictionary where each key is a sample name, and the corresponding value is the downstream
            prediction for that sample site.
        """
        predictions: ElementData = {}
        for sample_name, data in nx_items(self.sample_network):
            predictions[sample_name] = cast(
                # pyre-fixme[16]: Item `float` of `Union[Expression, float]` has no
                #  attribute `value`.
                float,
                data.my_total_tracer_flux.value / data.my_total_flux.value,
            )
        return predictions

    def get_upstream_prediction_dictionary(self) -> ElementData:
        """
        Returns the upstream predictions as a dictionary.

        This method returns a dictionary containing the upstream predictions for each sample site in the network.
        The keys in the dictionary represent the sample names, and the corresponding values represent the upstream
        predictions for each sample site.

        Returns:
            A dictionary where each key is a sample name, and the corresponding value is the upstream
            prediction for that sample site.
        """
        # Get the predicted upstream concentration we found
        predictions: ElementData = {}
        for sample_name, data in nx_items(self.sample_network):
            # pyre-fixme[16]: `Optional` has no attribute `value`.
            predictions[sample_name] = data.my_tracer_value.value
        return predictions

    def get_misfit(self) -> float:
        """
        Returns the misfit value.

        This method returns the misfit value, which represents the discrepancy between the observed data
        and the model predictions. The misfit value is calculated as the norm of the stacked primary terms.

        Returns:
            float: The misfit value.
        """
        return cp.norm(cp.vstack(self._primary_terms)).value

    def get_roughness(self) -> float:
        """
        Returns the roughness value.

        This method returns the total size of the regularization term in the optimization problem. This corresponds
        to the total deviation of the upstream concentrations from the geometric mean of the compositions.

        Returns:
            float: The roughness value.
        """
        return cp.norm(cp.vstack(self._regularizer_terms)).value


def get_element_obs(element: str, obs_data: pd.DataFrame) -> ElementData:
    """
    Extracts observed element data from a pandas DataFrame.

    Args:
        element: The name of the element for which the data is to be extracted.
        obs_data: The pandas DataFrame containing the observed element data.

    Returns:
        ElementData: A dictionary containing the observed element data, where the keys are sample names and the values
            are the corresponding observed element concentrations.
    """
    element_data: ElementData = {
        e: c
        # pyre-fixme[29]: `Series` is not a function.
        for e, c in zip(obs_data[SAMPLE_CODE_COL_NAME].tolist(), obs_data[element].tolist())
        if isinstance(c, float)
    }
    return element_data


def forward_model(
    sample_network: nx.DiGraph,
    upstream_concentrations: ElementData,
    export_rates: Optional[ExportRateData] = None,
) -> ElementData:
    """Predicts the downstream concentration at sample sites using a forward model.
    Args:
        sample_network: A sample_network of localities (see `get_sample_graphs`)
        upstream_concentrations: Dictionary of upstream concentrations at sample sites
        export_rates: Dictionary of export rates for each sub-catchment. Defaults to equal export rate in each sub-catchment.
    Returns:
        mixed_downstream_pred: Dictionary containing predicted downstream mixed concentration at each sample sites
    """
    mixed_downstream_pred: ElementData = {}

    for data in nx_values(sample_network):
        data.my_total_tracer_flux = 0.0
        data.my_total_flux = 0.0

    for sample_name, my_data in nx_topological_sort_with_data(sample_network):
        # If provided, set export rates from user input
        # else default to equal rate (absolute value is arbitrary)
        # pyre-fixme[8]: Attribute has type `Parameter`; used as `float`.
        my_data.my_export_rate = export_rates[sample_name] if export_rates else 1.0

        # pyre-fixme[8]: Attribute has type `Optional[Expression]`; used as `float`.
        my_data.my_tracer_value = upstream_concentrations[sample_name]
        # area weighted total contribution of material from this node
        my_data.my_flux = my_data.area * my_data.my_export_rate
        # Add the flux I generate to the total flux passing through me
        my_data.my_total_flux += my_data.my_flux
        # area weighted contribution of *tracer* from this node
        # pyre-fixme[58]: `*` is not supported for operand types `Union[None,
        #  cp.expressions.expression.Expression, float]` and
        #  `Optional[cp.expressions.expression.Expression]`.
        my_data.my_tracer_flux = my_data.my_flux * my_data.my_tracer_value
        # Add the *tracer* flux I generate to the total flux of *tracer* passing through me
        my_data.my_total_tracer_flux += my_data.my_tracer_flux

        normalised = my_data.my_total_tracer_flux / my_data.my_total_flux
        mixed_downstream_pred[sample_name] = normalised
        if (ds := nx_get_downstream_data(sample_network, sample_name)) is not None:
            # Add our flux to downstream node's
            ds.my_total_flux += my_data.my_total_flux
            # Add our *tracer* flux to the downstream node's
            ds.my_total_tracer_flux += my_data.my_total_tracer_flux

    return mixed_downstream_pred


def mix_downstream(
    sample_network: nx.DiGraph,
    areas: Dict[str, npt.NDArray[np.float_]],
    concentration_map: npt.NDArray[np.float_],
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
    # Calculate average concentration in each area:
    mixed_upstream_pred = {
        sample_name: float(np.mean(concentration_map[area])) for sample_name, area in areas.items()
    }
    # Predict downstream concentration at each sample site
    mixed_downstream_pred = forward_model(sample_network, mixed_upstream_pred, export_rates)
    return mixed_downstream_pred, mixed_upstream_pred


def get_unique_upstream_areas(
    sample_network: nx.DiGraph,
) -> Dict[str, npt.NDArray[np.bool_]]:
    """
    Generates a dictionary mapping sample numbers to unique upstream areas as boolean masks.

    Args:
        sample_network: The network of sample sites along the drainage, with associated data.

    Returns:
        A dictionary where the keys are sample numbers and the values are boolean masks
        representing the unique upstream areas for each sample site.

    Note:
        The function generates a dictionary that maps each sample number in the sample network onto a boolean mask
        representing the unique upstream area associated with that sample site. The boolean mask is obtained from an
        image file, assuming the presence of a file named "labels.tif" (generated after calling `get_sample_graphs).
        The pixel values in the image correspond to the labels of the unique upstream areas.

        The function reads the image file using `plt.imread()` and extracts the first channel (`[:, :, 0]`) as the
        labels. It then creates a boolean mask for each sample site by comparing the labels to the label of the sample
        site in the sample network data.

        The resulting dictionary provides a mapping between each sample number and its unique upstream area as a boolean
        mask.
    """
    I = plt.imread("labels.tif")[:, :, 0]
    return {node: I == data.label for node, data in nx_items(sample_network)}


def plot_sweep_of_regularizer_strength(
    sample_network: SampleNetworkUnmixer,
    element_data: ElementData,
    min_: float,
    max_: float,
    trial_num: int,
) -> None:
    """
    Plot a sweep of regularization strengths and their impact on roughness and data misfit.

    Args:
        sample_network: The network of sample sites along the drainage, with associated data.
        element_data: Dictionary of element data.
        min_: The minimum exponent for the logspace range of regularization strengths to try.
        max_: The maximum exponent for the logspace range of regularization strengths to try.
        trial_num: The number of regularization strengths to try within the specified range.

    Note:
        The function performs a sweep of regularization strengths within a specified logspace range and plots their
        impact on the roughness and data misfit of the sample network. For each regularization strength value, it
        solves the sample network problem using the specified solver ("ecos") and the corresponding regularization
        strength. It then calculates the roughness and data misfit values using the network's `get_roughness()` and
        `get_misfit()` methods, respectively.

        The roughness and data misfit values are plotted as a scatter plot, with the regularization strength value
        displayed as text next to each point. The x-axis represents the roughness values, and the y-axis represents the
        data misfit values.

        The function also prints the roughness and data misfit values for each regularization strength value.

        Finally, the function displays the scatter plot with appropriate axis labels.
    """
    vals = np.logspace(min_, max_, num=trial_num)  # regularizer strengths to try
    for val in tqdm.tqdm(vals, total=len(vals)):
        _ = sample_network.solve(element_data, solver="ecos", regularization_strength=val)
        roughness = sample_network.get_roughness()
        misfit = sample_network.get_misfit()
        plt.scatter(roughness, misfit, c="grey")
        plt.text(roughness, misfit, str(round(np.log10(val), 3)))
    plt.xlabel("Roughness")
    plt.ylabel("Data misfit")
    plt.show()


def get_upstream_concentration_map(
    areas: Dict[str, npt.NDArray[np.bool_]],
    upstream_preds: Dict[str, float],
) -> npt.NDArray[np.float_]:
    """
    Generate a two-dimensional map displaying the predicted upstream concentration for a given element for each unique upstream area.

    Args:
        areas: Dictionary mapping sample numbers onto a boolean mask representing the unique upstream areas.
        upstream_preds: Dictionary of predicted upstream concentrations.

    Returns:
        np.ndarray: A two-dimensional map displaying the predicted upstream concentration for each unique upstream area.

    Note:
        The function takes two inputs: `areas`, which is a dictionary mapping sample numbers to boolean masks representing
        the unique upstream areas, and `upstream_preds`, which is a dictionary of predicted upstream concentrations.

        The function initializes an output array (`out`) with the same shape as the boolean masks in `areas`. It then
        iterates over the sample numbers and corresponding predicted upstream concentrations in `upstream_preds`, and
        accumulates the concentrations in the respective areas of `out`.

        The resulting `out` array represents a two-dimensional map displaying the predicted upstream concentration for
        each unique upstream area.

    """

    out = np.zeros(list(areas.values())[0].shape)  # initialise output
    for sample_name, value in upstream_preds.items():
        out[areas[sample_name]] += value
    return out


def visualise_downstream(pred_dict: ElementData, obs_dict: ElementData, element: str) -> None:
    """
    Visualize the predicted downstream concentrations against the observed concentrations for a given element.

    Args:
        pred_dict: Predicted downstream concentrations.
        obs_dict: Observed downstream concentrations.
        element: The symbol of the element.

    Note:
        The function takes three inputs: `pred_dict`, which is a dictionary of predicted downstream concentrations,
        `obs_dict`, which is a dictionary of observed downstream concentrations, and `element`, which is the symbol
        of the element.

        The function retrieves the observed and predicted concentrations from the dictionaries and plots them as a
        scatter plot. The x-axis represents the observed concentrations, and the y-axis represents the predicted
        concentrations. The plot is displayed with logarithmic scaling on both axes.

        Additionally, a diagonal line is plotted as a reference, and the axis limits are set to show the data points
        without excessive padding. The aspect ratio of the plot is set to 1.
    """

    # Loop through keys in obs_pred and extract values of element.
    # Store the observed and predicted values in separate np arrays
    obs = []
    pred = []
    for sample in obs_dict:
        # NOTE: Dicts have different orders, so we can't use a comprehension
        obs.append(obs_dict[sample])
        pred.append(pred_dict[sample])
    obs = np.array(obs)
    pred = np.array(pred)
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
