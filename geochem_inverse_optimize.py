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
            # TODO(alexlipp,r-barnes): Make sure that his uses the same summation strategy as the primary terms
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
    ) -> Tuple[ElementData, ElementData]:
        obs_mean: float = np.mean(list(observation_data.values()))

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
        downstream_preds = self.get_downstream_prediction_dictionary()
        upstream_preds = self.get_upstream_prediction_dictionary()

        downstream_preds = {sample: value * obs_mean for sample, value in downstream_preds.items()}
        upstream_preds = {sample: value * obs_mean for sample, value in upstream_preds.items()}

        return downstream_preds, upstream_preds

    def get_downstream_prediction_dictionary(self) -> ElementData:
        # Print the solution we found
        predictions: ElementData = {}
        for sample_name, data in self.sample_network.nodes(data=True):
            data = data["data"]
            predictions[sample_name] = data.total_flux.value / data.total_upstream_area

        return predictions

    def get_upstream_prediction_dictionary(self) -> ElementData:
        # Get the predicted upstream concentration we found
        predictions: ElementData = {}
        for sample_name, data in self.sample_network.nodes(data=True):
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


def get_unique_upstream_areas(sample_network: nx.DiGraph) -> Dict[str, np.ndarray]:
    """Generates a dictionary which maps sample numbers onto
    the unique upstream area (as a boolean mask)
    for the sample site."""
    I = plt.imread("labels.tif")[:, :, 0]
    return {node: I == data["data"].label for node, data in sample_network.nodes(data=True)}


def calibrate_regularizer(
    sample_network: nx.DiGraph,
    element_data: ElementData,
    min_: float,
    max_: float,
    number: float,
):
    vals = np.logspace(min_, max_, num=number)  # regularizer strengths to try
    for val in vals:
        print("________________________________________________________")
        print("Trying regularizer strength: 10^", round(np.log10(val), 3))
        _, _ = sample_network.solve(element_data, solver="ecos", regularization_strength=val)
        roughness = cp.norm(cp.vstack(sample_network._regularizer_terms)).value
        misfit = cp.norm(cp.vstack(sample_network._primary_terms)).value
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
    data_dir: str, data_filename: str, excluded_elements: Optional[List[str]] = None
) -> pd.DataFrame:
    sample_network, sample_adjacency = get_sample_graphs(data_dir)

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
        data_dir="data/",
        data_filename="data/geochem_no_dupes.dat",
        excluded_elements=["Bi", "S"],
    )
    print(results)


if __name__ == "__main__":
    main()
