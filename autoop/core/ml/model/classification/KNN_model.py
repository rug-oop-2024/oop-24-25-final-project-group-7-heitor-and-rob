from autoop.core.ml.model.model import Model
import numpy as np
import pandas as pd
from typing import Any
from copy import deepcopy


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors classifier.

    Attributes:
        k (int): Number of neighbors to use.
        name (str): Name of the model.
        type (str): Type of the model.
        observations (np.ndarray): Training observations.
        ground_truth (np.ndarray): Ground truth labels.
        _parameters (dict): Model parameters.
    """

    def __init__(self, k: int = 3, name: str = "K-Nearest Neighbors", type: str = "classification") -> None:
        """
        Initialize the KNearestNeighbors model.

        Args:
            k (int): Number of neighbors to use.
            name (str): Name of the model.
            type (str): Type of the model.
        """
        super().__init__(name=name, type=type)
        self.k = k
        self.observations = None
        self.ground_truth = None
        self._parameters = {}

    @property
    def parameters(self) -> dict:
        """
        Get the model parameters.

        Returns:
            dict: A dictionary of model parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """
        Set the model parameters.

        Args:
            value (dict): A dictionary of model parameters.

        Raises:
            ValueError: If the provided value is not a dictionary.
        """
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model using the provided observations and ground truth labels.

        Args:
            observations (np.ndarray): Training observations.
            ground_truth (np.ndarray): Ground truth labels.

        Raises:
            ValueError: If the number of observations does not match the number of ground truth labels.
        """
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                "The number of observations must match the number of ground truth labels.")

        self.observations = observations
        self.ground_truth = ground_truth
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the provided observations.

        Args:
            observations (np.ndarray): Observations to predict.

        Returns:
            np.ndarray: Predicted labels.
        """
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> Any:
        """
        Predict the label for a single observation.

        Args:
            observation (np.ndarray): A single observation to predict.

        Returns:
            Any: Predicted label.
        """
        distances = np.linalg.norm(self.observations - observation, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.ground_truth[k_indices]
        most_common = pd.Series(k_nearest_labels).value_counts()
        return most_common.idxmax()
