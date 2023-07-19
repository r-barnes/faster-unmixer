#!/usr/bin/env python
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from cvxpy.error import SolverError
from scipy.optimize import minimize

from composition import Composition, composition_from_clr
from sample_network_unmix import (ElementData, SampleNetworkUnmixer,
                                  get_element_obs)

MultiElementData = Dict[str, ElementData]
"""Type for suite of multiple tracers"""


def get_multielementdata(obs_data: pd.DataFrame, elements: List[str]) -> MultiElementData:
    """
    Returns dictionary of {element: {sample_name: concentration}}
    """
    multielement_data = {}
    for element in elements:
        element_data = get_element_obs(element, obs_data)
        multielement_data[element] = element_data
    return multielement_data


def dict_to_array(dictionary: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Convert a dictionary into a NumPy array.

    Args:
        dictionary (Dict[str, Any]): The dictionary to convert.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing the NumPy array and the list of keys.
    """
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    return np.array(values), keys


def array_to_dict(array: np.ndarray, keys: List[str]) -> Dict[str, Any]:
    """Convert a NumPy array and a list of keys into a dictionary.

    Args:
        array (np.ndarray): The NumPy array.
        keys (List[str]): The list of keys.

    Returns:
        Dict[str, Any]: The resulting dictionary.
    """
    dictionary = {}
    for i, key in enumerate(keys):
        dictionary[key] = array[i]
    return dictionary


def has_one_sink_node(graph: nx.DiGraph) -> bool:
    """Check if a networkx DiGraph has exactly one sink node.

    Args:
        graph (nx.DiGraph): The directed graph to check.

    Returns:
        bool: True if the graph has exactly one sink node, False otherwise.
    """
    sink_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
    return len(sink_nodes) == 1


class ExportRateOptimizer:
    """Takes nested sample network of conservative well mixed tracers
        and inverts to return upstream concentration for each tracer and the
        best fitting export rate (e.g., sediment generation, run-off) of each node.

        Attributes:
            regulariser_strength (float): The strength of the regularizer.

        Property methods:
            data_misfit (float): The global misfit given tracer observations and export rates
            model_size (float): The size of the export rates model
            objective (float): The total size of the objective function
    _
        Methods:
            optimise: Optimize the export rates using the Powell method.

    """

    def __init__(
        self,
        source_optimiser: SampleNetworkUnmixer,
        observations: MultiElementData,
        export_regulariser_strength: float,
        source_regulariser_strength: float,
    ) -> None:
        """
        Initialize the ExportRateOptimizer.

        Args:
            source_optimiser (SampleNetworkUnmixer): The nested sample network of conservative well mixed tracers.
            observations (MultiElementData): The tracer observations.
            regulariser_strength (float): The strength of the export rate regularizer.
            source_regulariser_strength (float): The strength of the export rate regulariser.

        Raises:
            ValueError: If the sample network does not have exactly one root node.
            ValueError: If the nodes in the sample network do not match the nodes in the observations.
        """
        if not has_one_sink_node(source_optimiser.sample_network):
            raise ValueError("Sample Network must have exactly one root node")

        if set(source_optimiser.sample_network.nodes) != set(
            observations[next(iter(observations))]
        ):
            raise ValueError("Nodes in SampleNetworkUnmixer do not match nodes in observations")

        self._inverse_problem: SampleNetworkUnmixer = source_optimiser
        self._nodes: List[str] = list(self._inverse_problem.sample_network.nodes)
        self._tracer_observations: MultiElementData = observations
        self.regulariser_strength = export_regulariser_strength
        self.source_regulariser = source_regulariser_strength
        # Initially assume equal export rates (i.e. a clr vector of zero's)
        self._export_rates: Composition = composition_from_clr({node: 0 for node in self._nodes})

    def _calculate_objective(self, trial_export_rate_clrs: np.ndarray) -> float:
        """
        Compute the objective function value for a set of trial export rates.

        Args:
            trial_export_rate_clrs (np.ndarray): Trial export rates represented as an array of clr parameters.

        Returns:
            float: The objective function value.
        """
        parameter_dict = array_to_dict(array=trial_export_rate_clrs, keys=self._nodes)
        self._export_rates = composition_from_clr(parameter_dict)
        self._objective = self.data_misfit + self.regulariser_strength * self.model_size
        return self.objective

    @property
    def objective(self) -> float:
        """
        Compute the total objective function considering model size and data misfit

        Returns:
            float: Objective function
        """
        # print(self._objective)
        return self._objective

    @property
    def model_size(self) -> float:
        """
        Compute the norm of the export rate vector.

        Returns:
            float: The norm of the export rates clr vector
        """
        return self.export_rates.norm

    @property
    def data_misfit(self) -> float:
        """
        Compute the data misfit for the tracer observations.

        Returns:
            float: The data misfit value.
        """
        misfit = 0
        for observations in self._tracer_observations.values():
            try:
                _, _ = self._inverse_problem.solve(
                    observations,
                    solver="ecos",
                    export_rates=self._export_rates.composition,
                    regularization_strength=self.source_regulariser,
                )
            except SolverError:
                print("\n ECOS solver failed, trying again with SCS solver... \n")
                _, _ = self._inverse_problem.solve(
                    observations,
                    solver="scs",
                    export_rates=self._export_rates.composition,
                    regularization_strength=self.source_regulariser,
                )
            # Get the squared relative difference. Subtract n_samps so that minimum is 0.
            misfit += self._inverse_problem.get_misfit()
        return misfit

    @property
    def export_rates(self) -> Composition:
        """
        Get the current export rates.

        Returns:
            Composition: The optimized export rates.
        """
        return self._export_rates

    @export_rates.setter
    def export_rates(self, new_export_rates: Composition) -> None:
        """
        Set the export rates.
        """
        self._export_rates = new_export_rates

    def optimise(self) -> None:
        """
        Optimizes the export rates using Powell's method.
        """
        initial_clr, _ = dict_to_array(self.export_rates.clr)
        result = minimize(
            fun=self._calculate_objective,
            x0=initial_clr,
            method="powell",
            options={"xtol": 1e-3, "ftol": 1e-2},
        )
        if result.success:
            print("!! Success !!")
            print(self.export_rates.composition)
        else:
            print("Optimization failed!")
