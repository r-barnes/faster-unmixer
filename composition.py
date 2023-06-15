"""
This module provides classes and functions for working with compositional data.

Classes:
- Composition: Represents a compositional data point and provides operations for CLR transform.
- CompositionalDataset: Represents a dataset of more than one Compositions.

Functions:
- aitchison_distance(a: Composition, b: Composition) -> float: Calculates Aitchison distance between two Compositions.
- add(a: Composition, b: Composition) -> Composition: Performs the addition operation a + bon two Compositions *on the Simplex*
- subtract(a: Composition, b: Composition) -> Composition: Performs the subtraction operation a - b on two Compositions *on the Simplex*
- multiply(a: Composition, b: Composition) -> Composition: Performs the scalar multiplication operation a*k between Composition and a scalar on the Simplex*

"""

import math
from typing import Dict, Set

import pandas as pd


class Composition:
    """
    Represents a compositional data point and provides operations for CLR transform .

    Attributes:
    - composition (Dict[str, float]): The composition as a dictionary of component names and corresponding values.
    - clr (Dict[str, float]): The centred log ratio (CLR) transformation of the composition.
    - components (Set[str]): Set of the component names in the composition
    - norm (float): Size of the composition (Aitchison distance to neutral composition)

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
        self._components: Set[str] = set(composition.keys())

    @property
    def composition(self) -> Dict[str, float]:
        """Getter property for the composition."""
        return self._composition

    @composition.setter
    def composition(self, new_composition: Dict[str, float]) -> None:
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
        self._components = set(new_composition.keys())

    @property
    def clr(self) -> Dict[str, float]:
        """Getter property for the centred log ratio (CLR) transformation."""
        return self._clr

    @clr.setter
    def clr(self, new_clr: Dict[str, float]) -> None:
        """Setter property for the centred log ratio (CLR) transformation."""
        self._clr = new_clr
        self._composition = Composition.inverse_clr_transform(new_clr)
        self._components = set(new_clr.keys())

    @property
    def components(self) -> Set[str]:
        """Getter property for the components."""
        return self._components

    @property
    def norm(self) -> float:
        """Returns the norm of the CLR vector for composition."""
        squared_sum = sum(value**2 for value in self.clr.values())
        norm = math.sqrt(squared_sum)
        return norm

    @staticmethod
    def clr_transform(composition: Dict[str, float]) -> Dict[str, float]:
        """
        Performs the centred log ratio (CLR) transform on a compositional vector.

        Parameters:
        - composition (Dict[str, float]): The compositional vector to transform.

        Returns:
        - Dict[str, float]: The CLR transformation of the compositional vector.
        """
        geometric_mean = Composition.calculate_geometric_mean(composition)
        clr = {
            component: math.log(value / geometric_mean) for component, value in composition.items()
        }
        return clr

    @staticmethod
    def inverse_clr_transform(clr: Dict[str, float]) -> Dict[str, float]:
        """
        Performs the inverse centred log ratio (CLR) transformation on a CLR vector.

        Parameters:
        - clr (Dict[str, float]): The CLR vector to transform.

        Returns:
        - Dict[str, float]: The inverse CLR transformation of the CLR vector.
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


def aitchison_distance(a: Composition, b: Composition) -> float:
    """
    Calculates Aitchison distance between two Compositions.

    Parameters:
    - a (Composition): The first Composition.
    - b (Composition): The second Composition.

    Returns:
    - float: The Aitchison distance between the two Compositions.

    Raises:
    - ValueError: If the compositional components are not the same between the two Compositions.
    """
    a_clr, b_clr = a.clr, b.clr
    if a.components != b.components:
        raise ValueError("Compositional components are not the same")
    squared_diff_sum = sum((a_clr[key] - b_clr[key]) ** 2 for key in a_clr)
    aitchison_distance = math.sqrt(squared_diff_sum)

    return aitchison_distance


def add(a: Composition, b: Composition) -> Composition:
    """Performs the operation a + b in the Simplex.

    Args:
        a (Composition): The first composition.
        b (Composition): The second composition.

    Returns:
        Composition: The result of adding a and b.

    Raises:
        ValueError: If the compositional components are not the same.
    """
    a_clr, b_clr = a.clr, b.clr
    if a.components != b.components:
        raise ValueError("Compositional components are not the same")
    out_clr = {key: a_clr[key] + b_clr[key] for key in a_clr}
    return Composition(Composition.inverse_clr_transform(out_clr))


def subtract(a: Composition, b: Composition) -> Composition:
    """Performs the operation a - b in the Simplex.

    Args:
        a (Composition): The composition to subtract from.
        b (Composition): The composition to subtract.

    Returns:
        Composition: The result of subtracting b from a.

    Raises:
        ValueError: If the compositional components are not the same.
    """
    a_clr, b_clr = a.clr, b.clr
    if a.components != b.components:
        raise ValueError("Compositional components are not the same")
    out_clr = {key: a_clr[key] - b_clr[key] for key in a_clr}
    return Composition(Composition.inverse_clr_transform(out_clr))


def multiply(a: Composition, k: float) -> Composition:
    """Performs the operation a * k in the Simplex where k is a scalar.

    Args:
        a (Composition): The composition.
        k (float): The scalar to multiply with.

    Returns:
        Composition: The result of multiplying a by k.
    """
    out_clr = {key: value * k for key, value in a.clr.items()}
    return Composition(Composition.inverse_clr_transform(out_clr))


def _check_identical_components(compositions: Dict[str, Composition]) -> bool:
    """
    Checks if every composition in a dictionary of compositions has an identical set of components.

    Args:
        compositions Dict[str,Composition]: The dictionary of compositions to check.

    Returns:
        bool: True if all compositions have identical components, False otherwise.
    """

    first_components = set(compositions[next(iter(compositions))].components)
    for composition in compositions.values():
        if set(composition.components) != first_components:
            return False
    return True


class CompositionalDataset:
    """
    Represents a dataset of more than one compositions.

    Attributes:
    - compositions (Dict[str, Composition]): The compositions in the dataset.
    - components (Set[str]): The components of the dataset

    Property methods:
    - geometric_mean: Returns the geometric mean of the Dataset (i.e., the centroid).
    - arithmetic_mean: Returns the arithmetic mean of the Dataset (i.e., the mixture).
    - clr_df: Returns the clr vectors of the dataset as a DataFrame.
    - composition_df: Returns the closed compositions of the dataset as a DataFrame.
    - components: Set of components

    Methods:
    - __init__(input_df: pd.DataFrame): Initializes a CompositionalDataset instance.
    - calculate_distance_matrix: Returns the Aitchison distance matrix between compositions in the dataset.

    """

    def __init__(self, compositions: Dict[str, Composition]) -> None:
        """
        Initializes a CompositionalDataset instance.

        Parameters:
        - input_df (pd.DataFrame): Dictionary of compositions.

        Raises:
        - ValueError: If the input DataFrame contains invalid values.
        """
        if _check_identical_components(compositions):
            self._compositions: Dict[str, Composition] = compositions
            self._components = set(compositions[next(iter(compositions))].components)
        else:
            raise ValueError("All Compositions must have same set of components")

    @property
    def compositions(self) -> Dict[str, Composition]:
        """Getter method for compositions"""
        return self._compositions

    @property
    def components(self) -> Set[str]:
        """Getter method for components"""
        return self._components

    @compositions.setter
    def compositions(self, new_compositions: Dict[str, Composition]) -> None:
        """Setter property for the compositions."""
        if _check_identical_components(new_compositions):
            self._compositions: Dict[str, Composition] = new_compositions
            self._components = set(new_compositions[next(iter(new_compositions))].components)
        else:
            raise ValueError("All Compositions must have same set of components")

    @property
    def clr_df(self) -> pd.DataFrame:
        """
        Property method: Returns the clr vectors of the dataset as a DataFrame.

        Returns:
        - pd.DataFrame: The clr vectors of the dataset.
        """
        return pd.DataFrame(
            {name: composition.clr for name, composition in self.compositions.items()}
        ).T

    @property
    def composition_df(self) -> pd.DataFrame:
        """
        Property method: Returns the closed compositions of the dataset as a DataFrame.

        Returns:
        - pd.DataFrame: The closed compositions of the dataset.
        """
        return pd.DataFrame(
            {name: composition.composition for name, composition in self.compositions.items()}
        ).T

    @property
    def geometric_mean(self) -> Composition:
        """
        Property method: Calculates the geometric mean of the Dataset (i.e., the centroid).

        Returns:
        - Composition: The geometric mean of the Dataset.
        """
        clr_mean = self.clr_df.mean().to_dict()
        return Composition(Composition.inverse_clr_transform(clr_mean))

    @property
    def arithmetic_mean(self) -> Composition:
        """
        Property method: Calculates the arithmetic mean of the Dataset (i.e., the mixture).

        Returns:
        - Composition: The arithmetic mean of the Dataset.
        """
        comp_mean = self.composition_df.mean().to_dict()
        return Composition(comp_mean)

    def calculate_distance_matrix(self) -> pd.DataFrame:
        """
        Returns the Aitchison distance matrix between compositions in the dataset.

        Returns:
        - pd.DataFrame: The Aitchison distance matrix.
        """
        compositions = list(self.compositions.values())
        composition_names = list(self.compositions.keys())
        num_compositions = len(compositions)

        distance_matrix = pd.DataFrame(index=composition_names, columns=composition_names)

        for i in range(num_compositions):
            for j in range(i, num_compositions):
                distance = aitchison_distance(compositions[i], compositions[j])
                distance_matrix.iloc[i, j] = distance
                distance_matrix.iloc[j, i] = distance

        return distance_matrix


def compositions_from_dataframe(input_df: pd.DataFrame) -> CompositionalDataset:
    """Constructs a CompositionalDataset from a pandas dataframe"""
    if input_df.isna().any().any():
        raise ValueError("Compositional datasets cannot contain NaN values")
    if input_df.isnull().any().any():
        raise ValueError("Compositional datasets cannot contain Null values")
    if (input_df <= 0).any().any():
        raise ValueError("Compositional datasets must contain only strictly positive values")
    if input_df.apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise ValueError("Compositional datasets must only contain numerics")
    if input_df.shape[0] <= 1:
        raise ValueError("CompositionalDatasets must have more than one row")

    compositions_dict = {
        row_index: Composition(row.to_dict()) for row_index, row in input_df.iterrows()
    }
    return CompositionalDataset(compositions_dict)
