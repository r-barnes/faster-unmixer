"""
Module: composition.py

Provides a class for representing and manipulating a compositional data point.

Classes:
- Composition: Represents a compositional data point and provides operations for transformation and analysis.

"""


import math
from typing import Dict


class Composition:
    """
    Represents a compositional data point and provides operations for transformation and analysis.

    Attributes:
    - composition (Dict[str, float]): The composition as a dictionary of component names and corresponding values.
    - clr (Dict[str, float]): The centred log ratio (CLR) transformation of the composition.

    Methods:
    - __init__(self, composition: Dict[str, float]): Initializes a Composition instance.
    - clr_transform(self, composition: Dict[str, float]) -> Dict[str, float]: Performs the CLR transformation on a composition.
    - inverse_clr_transform(self, clr: Dict[str, float]) -> Dict[str, float]: Performs the inverse CLR transformation on a CLR composition.
    - calculate_geometric_mean(self, data: Dict[str, float]) -> float: Calculates the geometric mean of a dictionary of values.
    """

    def __init__(self, composition: Dict[str, float]):
        """
        Initializes a Composition instance.

        Parameters:
        - composition (Dict[str, float]): A dictionary representing the composition, where the values must be strictly positive.

        Raises:
        - ValueError: If any of the compositional values are not strictly positive.
        """
        if any(value < 0 for value in composition.values()):
            raise ValueError("Compositional values must be strictly positive")
        total_sum = sum(composition.values())
        self._composition = {key: value / total_sum for key, value in composition.items()}
        self._clr = Composition.clr_transform(composition)

    @property
    def composition(self) -> Dict[str, float]:
        """Getter property for the composition."""
        return self._composition

    @composition.setter
    def composition(self, new_composition: Dict[str, float]):
        """
        Setter property for the composition.

        Parameters:
        - new_composition (Dict[str, float]): The new composition to set.

        Raises:
        - ValueError: If any of the compositional values in the new composition are not strictly positive.
        """
        if any(value < 0 for value in new_composition.values()):
            raise ValueError("Compositional values must be strictly positive")
        total_sum = sum(new_composition.values())
        self._composition = {key: value / total_sum for key, value in new_composition.items()}
        self._clr = Composition.clr_transform(new_composition)

    @property
    def clr(self) -> Dict[str, float]:
        """Getter property for the centred log ratio (CLR) transformation."""
        return self._clr

    @clr.setter
    def clr(self, new_clr: Dict[str, float]):
        """Setter property for the centred log ratio (CLR) transformation."""
        self._clr = new_clr
        self._composition = Composition.inverse_clr_transform(new_clr)

    @staticmethod
    def clr_transform(composition: Dict[str, float]) -> Dict[str, float]:
        """
        Performs the centred log ratio (CLR) transform on a composition.

        Parameters:
        - composition (Dict[str, float]): The composition to transform.

        Returns:
        - Dict[str, float]: The CLR transformation of the composition.
        """
        geometric_mean = Composition.calculate_geometric_mean(composition)
        clr = {
            component: math.log(value / geometric_mean) for component, value in composition.items()
        }
        return clr

    @staticmethod
    def inverse_clr_transform(clr: Dict[str, float]) -> Dict[str, float]:
        """
        Performs the inverse centred log ratio (CLR) transformation on a CLR composition.

        Parameters:
        - clr (Dict[str, float]): The CLR composition to transform.

        Returns:
        - Dict[str, float]: The inverse CLR transformation of the CLR composition.
        """
        exp_values = {component: math.exp(clr_value) for component, clr_value in clr.items()}
        total_sum = sum(exp_values.values())
        composition = {key: value / total_sum for key, value in exp_values.items()}
        return composition

    @staticmethod
    def calculate_geometric_mean(data: Dict[str, float]) -> float:
        """
        Calculates the geometric mean of a dictionary of values.

        Parameters:
        - data (Dict[str, float]): The dictionary of values.

        Returns:
        - float: The geometric mean of the values.
        """
        product = 1
        count = len(data)

        for value in data.values():
            product *= value

        return math.pow(product, 1 / count)
