#!/usr/bin/env python3

import os
import tempfile
from typing import Any, Dict, Final, List, Optional, Tuple

# TODO(rbarnes): Make a requirements file for conda
import cvxpy as cp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyfastunmix

NO_DOWNSTREAM: Final[int] = 0
SAMPLE_CODE: Final[str] = "Sample.Code"
ELEMENT_LIST: Final[List[str]] = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]

ElementData = Dict[str, float]


def cp_log_ratio_norm(a, b):
    return cp.maximum(a / b, b * cp.inv_pos(a))


def nx_topological_sort_with_data(G: nx.DiGraph):
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


def get_sample_graphs(data_dir: str) -> Tuple[nx.DiGraph, Any]:
    # Get the graph representations of the data
    sample_network_raw, sample_adjacency = pyfastunmix.fastunmix(data_dir)

    ids_to_names: Dict[int, str] = {
        i: data.data.name for i, data in enumerate(sample_network_raw)
    }

    # Convert it into a networkx graph for easy use in Python
    sample_network = nx.DiGraph()
    for x in sample_network_raw[1:]:  # Skip the first node into which it all flows
        sample_network.add_node(x.data.name, data=x)
        if x.downstream_node != NO_DOWNSTREAM:
            sample_network.add_edge(x.data.name, ids_to_names[x.downstream_node])

    # Calculate the total contributing area for each sample
    for x, my_data in nx_topological_sort_with_data(sample_network):
        my_data.total_area += my_data.area
        if ds := nx_get_downstream(sample_network, x):
            downstream_data = sample_network.nodes[ds]["data"]
            downstream_data.total_area += my_data.total_area

    return sample_network, sample_adjacency


def reset_sample_network(sample_network: nx.DiGraph) -> None:
    for _, data in sample_network.nodes(data=True):
        data["data"].my_value = 0.0
        data["data"].my_flux = 0.0
        data["data"].total_flux = 0.0


def get_primary_terms(sample_network: nx.DiGraph, obs_data: ElementData) -> List[Any]:
    # Build the main objective
    # Use a topological sort to ensure an upstream-to-downstream traversal
    primary_terms = []
    for sample_name, my_data in nx_topological_sort_with_data(sample_network):
        # Set up a CVXPY parameter for each element for each node
        my_data.my_value = cp.Variable(pos=True)

        # area weighted contribution from this node
        # TODO(rbarnes): Try scaling area of a region by dividing by total area of all regions. `my_data.normalized_area`
        my_data.my_flux = my_data.area * my_data.my_value

        # Add the flux I generate to the total flux passing through me
        my_data.total_flux += my_data.my_flux
        obs_mean = np.mean(list(obs_data.values()))

        observed = (
            obs_data[my_data.data.name] / obs_mean
        )  # Normalise observation by mean
        normalised_concentration = my_data.total_flux / my_data.total_area
        primary_terms.append(cp_log_ratio_norm(normalised_concentration, observed))

        if ds := nx_get_downstream(sample_network, sample_name):
            downstream_data = sample_network.nodes[ds]["data"]
            # Add our flux to the downstream node's
            downstream_data.total_flux += my_data.total_flux

    return primary_terms


def get_regularizer_terms(sample_network: nx.DiGraph, adjacency_graph) -> List[Any]:
    # Build the regularizer
    regularizer_terms = []
    # for adjacent_nodes, border_length in sample_adjacency.items():
    #   node_a, node_b = adjacent_nodes
    #   a_data = sample_network.nodes[node_a]['data']
    #   b_data = sample_network.nodes[node_b]['data']
    #   for e in a_data.my_concentrations.keys():
    #     assert e in b_data.my_concentrations.keys()
    #     a_concen = a_data.my_concentrations[e]
    #     b_concen = b_data.my_concentrations[e]
    # TODO(r-barnes) replace regularizer misfit with log-ratio substitute
    #     regularizer_terms.append(border_length * (a_concen-b_concen))
    return regularizer_terms


def get_prediction_dictionary(sample_network: nx.DiGraph) -> pd.DataFrame:
    # Print the solution we found
    predictions: ElementData = {}
    for sample_name, data in sample_network.nodes(data=True):
        data = data["data"]
        predictions[sample_name] = data.total_flux.value / data.total_area

    return predictions


# TODO(rbarnes): Might need a per-element lambda value for the regularizer to find the elbow
def process_element(
    sample_network: nx.DiGraph,
    sample_adjacency: Any,
    obs_data: ElementData,
    regularizer_strength: float = 1e-3,
) -> ElementData:
    # Make a deep copy to avoid over-writing the original data
    reset_sample_network(sample_network)
    primary_terms = get_primary_terms(sample_network=sample_network, obs_data=obs_data)

    regularizer_terms = get_regularizer_terms(
        sample_network=sample_network, adjacency_graph=sample_adjacency
    )
    if not regularizer_terms:
        print("WARNING: No regularizer terms found!")

    # Build the objective and constraints
    objective = cp.norm(cp.vstack(primary_terms))
    if regularizer_terms:
        # TODO(alexlipp,r-barnes): Make sure that his uses the same summation strategy as the primary terms
        objective += regularizer_strength * cp.norm(cp.vstack(regularizer_terms))
    constraints = []

    # Create and solve the problem
    print("Compiling and solving problem...")
    problem = cp.Problem(cp.Minimize(objective), constraints)

    # Solvers that can handle this problem type include:
    # ECOS, SCS
    # See: https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
    # See: https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    solvers = {
        "scip": {"solver": cp.SCIP, "verbose": True},  # VERY SLOW, probably don't use
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
    objective_value = problem.solve(**solvers["ecos"])
    print(
        "{color}Status = {status}\033[39m".format(
            color="" if problem.status == "optimal" else "\033[91m",
            status=problem.status,
        )
    )
    print(f"Objective value = {objective_value}")
    obs_mean = np.mean(list(obs_data.values()))
    # Return outputs
    downstream_preds = get_prediction_dictionary(sample_network=sample_network)
    downstream_preds.update(
        (sample, value * obs_mean) for sample, value in downstream_preds.items()
    )

    return downstream_preds


def get_element_obs(element: str, obs_data: pd.DataFrame) -> ElementData:
    element_data: ElementData = {
        e: c
        for e, c in zip(obs_data[SAMPLE_CODE].tolist(), obs_data[element].tolist())
        if isinstance(c, float)
    }
    return element_data


def process_data(
    data_dir: str, data_filename: str, excluded_elements: Optional[List[str]] = None
) -> pd.DataFrame:
    sample_network, sample_adjacency = get_sample_graphs(data_dir)

    plot_network(sample_network)
    obs_data = pd.read_csv(data_filename, delimiter=" ")
    obs_data = obs_data.drop(columns=excluded_elements)

    results = None
    # TODO(r-barnes,alexlipp): Loop over all elements once we achieve acceptable results
    for element in ELEMENT_LIST[0:20]:
        if element not in obs_data.columns:
            continue

        print(f"\n\033[94mProcessing element '{element}'...\033[39m")

        # TODO(rbarnes): remove the `isinstance`
        element_data: ElementData = {
            e: c
            for e, c in zip(obs_data[SAMPLE_CODE].tolist(), obs_data[element].tolist())
            if isinstance(c, float)
        }

        try:
            predictions = process_element(
                sample_network=sample_network,
                sample_adjacency=sample_adjacency,
                obs_data=element_data,
            )
        except cp.error.SolverError as err:
            print(f"\033[91mSolver Error - skipping this element!\n{err}")
            continue

        obs = [element_data[sample] for sample in element_data.keys()]
        pred = [predictions[sample] for sample in element_data.keys()]

        if results is None:
            results = pd.DataFrame(element_data.keys())
        results[element + "_obs"] = obs
        results[element + "_dwnst_prd"] = pred

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
