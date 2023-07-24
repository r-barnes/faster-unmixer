# Add build to path
import sys

sys.path.append("../")
from typing import List

import networkx as nx
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sample_network_unmix as snu


class SampleNode:
    def __init__(self, name: str, area: float, downstream_node, x: int = None, y: int = None):
        self.name = name
        self.x = x
        self.y = y
        self.downstream_node = downstream_node
        self.area = area
        self.my_value = None
        self.my_flux = None
        self.total_flux = None


def generate_random_sample_network(n: int, areas: List[float], seed=None):
    """
    Generate a random sample network with n nodes with a random area between 0 and 1.
    """
    # Check that areas and N are compatible
    if len(areas) != n:
        raise ValueError("areas and N must be compatible")
    # Generate a random tree with n nodes
    G = nx.random_tree(n, create_using=nx.DiGraph, seed=seed)
    # Flip the tree upside down
    G = nx.reverse(G)
    # Loop through nodes and add SampleNode objects to ["data"] property
    i = 0
    for node in G.nodes:
        # Find downstream node of node
        downstream_node = snu.nx_get_downstream(G, node)
        G.nodes[node]["data"] = SampleNode(
            name=node,
            area=areas[i],
            downstream_node=downstream_node,
        )
        i += 1
    return G


def draw_random_log_uniform(min_val: float, max_val: float, n: int) -> np.ndarray:
    """
    Draws a sample from a log uniform distribution between min_val and max_val
    """
    # Raise error if min_val or max_val are negative
    if min_val < 0 or max_val < 0:
        raise ValueError("min_val and max_val must be positive")
    return np.exp(np.random.uniform(np.log(min_val), np.log(max_val), n))


def conc_list_to_dict(sample_network: nx.DiGraph, concs: np.ndarray) -> snu.ElementData:
    """
    Converts a list of concentrations to a dictionary of concentrations with keys corresponding to the node names in the
    sample network.
    """
    return {node: conc for node, conc in zip(sample_network.nodes, concs)}


def size_of_balanced_tree(r, h):
    """Calculate the number of nodes in a balanced tree with branching factor r and height h"""
    if r == 1:
        return h + 1
    return (r ** (h + 1) - 1) / (r - 1)


def generate_balanced_sample_network(
    branching_factor: int, height: int, areas: List[float]
) -> nx.DiGraph:
    """
    Generate a balanced sample network with branching factor branching_factor and height height.
    """
    # Check that areas and N are compatible
    if len(areas) != size_of_balanced_tree(branching_factor, height):
        raise ValueError("areas and network must be compatible")
    G = nx.balanced_tree(r=branching_factor, h=height, create_using=nx.DiGraph)
    G = G.reverse()
    # Loop through nodes and add SampleNode objects to ["data"] property
    i = 0
    for node in G.nodes:
        # Find downstream node of node
        downstream_node = snu.nx_get_downstream(G, node)
        G.nodes[node]["data"] = SampleNode(
            name=node,
            area=areas[i],
            downstream_node=downstream_node,
        )
        i += 1
    return G


def generate_r_ary_sample_network(branching_factor: int, N: int, areas: List[float]) -> nx.DiGraph:
    """
    Generate a full R-ary sample network with branching factor branching_factor and N nodes.
    """
    # Check that areas and N are compatible
    if len(areas) != N:
        raise ValueError("areas and N must be compatible")
    G = nx.full_rary_tree(r=branching_factor, n=N, create_using=nx.DiGraph)
    G = G.reverse()
    # Loop through nodes and add SampleNode objects to ["data"] property
    i = 0
    for node in G.nodes:
        # Find downstream node of node
        downstream_node = snu.nx_get_downstream(G, node)
        G.nodes[node]["data"] = SampleNode(
            name=node,
            area=areas[i],
            downstream_node=downstream_node,
        )
        i += 1
    return G


from hypothesis import given
from hypothesis import strategies as st


@given(
    N=st.integers(min_value=1, max_value=100),
    max_area=st.floats(min_value=0, max_value=1e6),
    min_area=st.floats(min_value=0, max_value=1e6),
    max_conc=st.floats(min_value=0, max_value=1e6),
    min_conc=st.floats(min_value=0, max_value=1e6),
)
def test_random_network(N: int, max_area: float, min_area: float, max_conc: float, min_conc: float):
    """
    Test that the SampleNetworkUnmixer can recover the upstream concentrations of a random sample network to tolerance of 0.1%.
    """
    # Check that min_area and min_conc are positive
    if min_area < 0 or min_conc < 0:
        raise ValueError("min_area and min_conc must be positive")
    # Check that max_area and max_conc are greater than min_area and min_conc respectively
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc
    # Check that N is positive
    if N < 0:
        raise ValueError("N must be positive")

    areas = draw_random_log_uniform(min_area, max_area, N)
    network = generate_random_sample_network(N, areas=areas, seed=0)
    concs_list = draw_random_log_uniform(min_area, max_conc, N)
    upstream = conc_list_to_dict(network, concs_list)
    downstream = snu.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = snu.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    _, recovered_upstream = problem.solve(downstream, solver="ecos")

    # Check that the recovered upstream concentrations are within 0.1% of the true upstream concentrations using
    # the np.isclose function
    for node in network.nodes:
        pred = recovered_upstream[node]
        true = upstream[node]
        assert np.isclose(pred, true, rtol=0.001)


@given(
    branching_factor=st.integers(min_value=1, max_value=5),
    height=st.integers(min_value=0, max_value=5),
    max_area=st.floats(min_value=0, max_value=1e6),
    min_area=st.floats(min_value=0, max_value=1e6),
    max_conc=st.floats(min_value=0, max_value=1e6),
    min_conc=st.floats(min_value=0, max_value=1e6),
)
def test_balanced_network(
    branching_factor: int,
    height: int,
    max_area: float,
    min_area: float,
    max_conc: float,
    min_conc: float,
):
    """
    Test that the SampleNetworkUnmixer can recover the upstream concentrations of a balanced sample network to tolerance of 0.1%.
    """
    # Check that min_area and min_conc are positive
    if min_area < 0 or min_conc < 0:
        raise ValueError("min_area and min_conc must be positive")
    # Check that max_area and max_conc are greater than min_area and min_conc respectively
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc
    # Check that the branching factor is greater than 1 and that the height is greater than 0
    if branching_factor < 1 or height < 0:
        raise ValueError(
            "branching_factor must be greater than 1 and height must be greater than 0"
        )

    N = int(size_of_balanced_tree(branching_factor, height))
    areas = draw_random_log_uniform(min_area, max_area, N)
    network = generate_balanced_sample_network(branching_factor, height, areas=areas)
    concs_list = draw_random_log_uniform(min_area, max_conc, N)
    upstream = conc_list_to_dict(network, concs_list)
    downstream = snu.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = snu.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    _, recovered_upstream = problem.solve(downstream, solver="ecos")

    # Check that the recovered upstream concentrations are within 0.1% of the true upstream concentrations using
    # the np.isclose function
    for node in network.nodes:
        pred = recovered_upstream[node]
        true = upstream[node]
        assert np.isclose(pred, true, rtol=0.001)


@given(
    branching_factor=st.integers(min_value=1, max_value=5),
    N=st.integers(min_value=0, max_value=100),
    max_area=st.floats(min_value=0, max_value=1e6),
    min_area=st.floats(min_value=0, max_value=1e6),
    max_conc=st.floats(min_value=0, max_value=1e6),
    min_conc=st.floats(min_value=0, max_value=1e6),
)
def test_rary_network(
    branching_factor: int,
    N: int,
    max_area: float,
    min_area: float,
    max_conc: float,
    min_conc: float,
):
    """
    Test that the SampleNetworkUnmixer can recover the upstream concentrations of a full R-ary sample network to tolerance of 0.1%.
    """
    # Check that min_area and min_conc are positive
    if min_area < 0 or min_conc < 0:
        raise ValueError("min_area and min_conc must be positive")
    # Check that max_area and max_conc are greater than min_area and min_conc respectively
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc
    # Check that the branching factor is greater than 1 and that the number of nodes is greater than 0
    if branching_factor < 1 or N < 0:
        raise ValueError("branching_factor must be greater than 1 and N must be greater than 0")

    areas = draw_random_log_uniform(min_area, max_area, N)
    network = generate_r_ary_sample_network(branching_factor, N, areas=areas)
    concs_list = draw_random_log_uniform(min_area, max_conc, N)
    upstream = conc_list_to_dict(network, concs_list)
    downstream = snu.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = snu.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    _, recovered_upstream = problem.solve(downstream, solver="ecos")

    # Check that the recovered upstream concentrations are within 0.1% of the true upstream concentrations using
    # the np.isclose function
    for node in network.nodes:
        pred = recovered_upstream[node]
        true = upstream[node]
        assert np.isclose(pred, true, rtol=0.001)
