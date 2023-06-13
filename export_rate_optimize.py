#!/usr/bin/env python3
from typing import Dict

from composition import Composition
from geochem_inverse_optimize import ElementData, SampleNetwork

# Type for suite of multiple tracers
MultiElementData = Dict[str, ElementData]


class ExportRateOptimizer:
    """Takes nested sample network of conservative well mixed tracers
    and inverts to return upstream concentration for each tracer and the
    best fitting export rate (e.g., sediment generation, run-off) of each node"""

    def __init__(
        self,
        source_optimiser: SampleNetwork,
        observations: MultiElementData,
        regulariser_strength: float,
    ) -> None:
        # Check number of root nodes. Raise exception if more than 1
        self.inverse_problem: SampleNetwork = source_optimiser
        # Check that the nodes in observations correspond to nodes in sample_network
        self.tracer_observations: MultiElementData = observations
        self.regulariser = regulariser_strength
        # Initiate this dictionary to have same labels but with equal values
        self.export_rates: Composition = Composition()
        self._data_misfit: float = None
        self._model_size: float = None
        self._objective: float = None

    # def _calculate_data_misfit() -> float:
    #   calculates and sets data misfit using current export rate values

    # def _calculate_model_size() -> float:
    #   calculates and sets model size using current export rate values

    # def _calculate_objective() -> float:
    #   calculates and sets objective function value using current export rate values

    def _make_default_export_rates(self) -> Dict[str, float]:
        # operate on self.sample_network to make dictionary of equal values
        return {}

    # def optimise() -> None:
    #   Uses Powells to optimise export rates

    @property
    def data_misfit(self):
        """Getter property for data misfit"""
        return self._data_misfit

    @property
    def model_size(self):
        """Getter property for model size"""
        return self._model_size

    @property
    def objective(self):
        """Getter property for objective function value"""
        return self._objective
