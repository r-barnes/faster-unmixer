# pyre-ignore-all-errors[56]

import math
from typing import Callable, Optional

import networkx as nx
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import funmixer


def max_height_of_balanced_tree(N: int, branching_factor: int) -> int:
    # Calculate the height of a balanced tree with N nodes and branching factor r
    # h = log_r(N(r-1)+1)-1
    return math.floor(math.log(N * (branching_factor - 1) + 1, branching_factor) - 1)


# Set the range to explore upstream concentration values over
MINIMUM_CONC = 1
MAXIMUM_CONC = 1e2

# Set the range to explore sub-basin area values over
MINIMUM_AREA = 1
MAXIMUM_AREA = 1e2

# Maximum number of nodes in a random network
MAXIMUM_NUMBER_OF_NODES = 100

# Maximum branching factor of a random network
MAXIMUM_BRANCHING_FACTOR = 4

# Maximum possible height of a balanced tree with given network parameters
MAXIMUM_HEIGHT: int = max_height_of_balanced_tree(MAXIMUM_NUMBER_OF_NODES, MAXIMUM_BRANCHING_FACTOR)

# Set the desired level of accuracy for tests to pass
TARGET_TOLERANCE = 0.01  # 0.01 = 1 %

# Set the solvers to try
SOLVERS = ["ecos", "gurobi"]


def draw_random_log_uniform(min_val: float, max_val: float) -> float:
    """
    Draws a sample from a log uniform distribution between min_val and max_val
    """
    # Raise error if min_val or max_val are negative
    assert min_val >= 0 and max_val >= 0
    return float(np.exp(np.random.uniform(np.log(min_val), np.log(max_val), 1))[0])


def conc_list_to_dict(
    sample_network: nx.DiGraph, concs: Callable[[], float]
) -> funmixer.ElementData:
    """
    Converts a list of concentrations to a dictionary of concentrations with keys corresponding to the node names in the
    sample network.
    """
    return {node: concs() for node in sample_network.nodes}


def size_of_balanced_tree(r: int, h: int) -> int:
    """Calculate the number of nodes in a balanced tree with branching factor r and height h"""
    if r == 1:
        return h + 1
    return (r ** (h + 1) - 1) / (r - 1)


def generate_random_sample_network(
    N: int, areas: Callable[[], float], seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Generate a random sample network with n nodes with a random area.
    """
    # Generate a random tree with n nodes
    G = nx.random_tree(N, create_using=nx.DiGraph, seed=seed)
    # Flip the tree upside down
    G = nx.reverse(G)
    # Loop through nodes and add SampleNode objects to ["data"] property
    i = 0
    for node in G.nodes:
        # Find downstream node of node
        G.nodes[node]["data"] = funmixer.SampleNode(
            name=node,
            area=areas(),
            downstream_node=funmixer.nx_get_downstream_node(G, node),
            x=-1,
            y=-1,
            total_upstream_area=0,
            label=0,
            upstream_nodes=[],
        )
        i += 1
    return G


def generate_balanced_sample_network(
    branching_factor: int, height: int, areas: Callable[[], float]
) -> nx.DiGraph:
    """
    Generate a balanced sample network with branching factor branching_factor and height height.
    """
    G = nx.balanced_tree(r=branching_factor, h=height, create_using=nx.DiGraph)
    G = G.reverse()
    # Loop through nodes and add SampleNode objects to ["data"] property
    i = 0
    for node in G.nodes:
        # Find downstream node of node
        G.nodes[node]["data"] = funmixer.SampleNode(
            name=node,
            area=areas(),
            downstream_node=funmixer.nx_get_downstream_node(G, node),
            x=-1,
            y=-1,
            total_upstream_area=0,
            label=0,
            upstream_nodes=[],
        )
        i += 1
    return G


def generate_r_ary_sample_network(
    N: int, branching_factor: int, areas: Callable[[], float]
) -> nx.DiGraph:
    """
    Generate a full R-ary sample network with branching factor branching_factor and N nodes.
    """
    # Check that areas and N are compatible
    G = nx.full_rary_tree(r=branching_factor, n=N, create_using=nx.DiGraph)
    G = G.reverse()
    # Loop through nodes and add SampleNode objects to ["data"] property
    i = 0
    for node in G.nodes:
        # Find downstream node of node
        G.nodes[node]["data"] = funmixer.SampleNode(
            name=node,
            area=areas(),
            downstream_node=funmixer.nx_get_downstream_node(G, node),
            x=-1,
            y=-1,
            total_upstream_area=0,
            label=0,
            upstream_nodes=[],
        )
        i += 1
    return G


# Explore random networks
@given(
    N=st.integers(min_value=2, max_value=MAXIMUM_NUMBER_OF_NODES),
    min_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    max_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    min_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    max_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    solver=st.sampled_from(SOLVERS),
)
@settings(deadline=None)
def test_random_network(
    N: int, min_area: float, max_area: float, min_conc: float, max_conc: float, solver: str
) -> None:
    """
    Test that the SampleNetworkUnmixer can recover the upstream concentrations of a random sample network to tolerance.
    """
    # Check that max_area and max_conc are greater than min_area and min_conc respectively
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc

    areas = lambda: draw_random_log_uniform(min_area, max_area)
    concentrations = lambda: draw_random_log_uniform(min_conc, max_conc)
    network = generate_random_sample_network(N=N, areas=areas)
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    solution = problem.solve(downstream, solver="ecos")

    # Check that the recovered upstream concentrations are within 0.1% of the true upstream concentrations using
    # the np.isclose function
    for node in network.nodes:
        pred = solution.upstream_preds[node]
        true = upstream[node]
        assert np.isclose(pred, true, rtol=TARGET_TOLERANCE)


# Explore balanced networks
@given(
    branching_factor=st.integers(min_value=1, max_value=MAXIMUM_BRANCHING_FACTOR),
    height=st.integers(min_value=1, max_value=MAXIMUM_HEIGHT),
    min_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    max_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    min_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    max_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    solver=st.sampled_from(SOLVERS),
)
@settings(deadline=None)
def test_balanced_network(
    branching_factor: int,
    height: int,
    min_area: float,
    max_area: float,
    min_conc: float,
    max_conc: float,
    solver: str,
) -> None:
    """
    Test that the SampleNetworkUnmixer can recover the upstream concentrations of a balanced sample network to tolerance.
    """
    # Check that max_area and max_conc are greater than min_area and min_conc respectively
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc

    areas = lambda: draw_random_log_uniform(min_area, max_area)
    concentrations = lambda: draw_random_log_uniform(min_conc, max_conc)
    network = generate_balanced_sample_network(
        branching_factor=branching_factor, height=height, areas=areas
    )
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    solution = problem.solve(downstream, solver="ecos")

    # Check that the recovered upstream concentrations are within 0.1% of the true upstream concentrations using
    # the np.isclose function
    for node in network.nodes:
        pred = solution.upstream_preds[node]
        true = upstream[node]
        assert np.isclose(pred, true, rtol=TARGET_TOLERANCE)


# Explore full r-ary networks
@given(
    branching_factor=st.integers(min_value=1, max_value=MAXIMUM_BRANCHING_FACTOR),
    N=st.integers(min_value=2, max_value=MAXIMUM_NUMBER_OF_NODES),
    min_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    max_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    min_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    max_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    solver=st.sampled_from(SOLVERS),
)
@settings(deadline=None)
def test_rary_network(
    branching_factor: int,
    N: int,
    min_area: float,
    max_area: float,
    min_conc: float,
    max_conc: float,
    solver: str,
) -> None:
    """
    Test that the SampleNetworkUnmixer can recover the upstream concentrations of a full R-ary sample network to tolerance.
    """
    # Check that max_area and max_conc are greater than min_area and min_conc respectively
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc

    areas = lambda: draw_random_log_uniform(min_area, max_area)
    concentrations = lambda: draw_random_log_uniform(min_conc, max_conc)
    network = generate_r_ary_sample_network(N=N, branching_factor=branching_factor, areas=areas)
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    solution = problem.solve(downstream, solver=solver)

    # Check that the recovered upstream concentrations are within 0.1% of the true upstream concentrations using
    # the np.isclose function
    for node in network.nodes:
        pred = solution.upstream_preds[node]
        true = upstream[node]
        assert np.isclose(pred, true, rtol=TARGET_TOLERANCE)
