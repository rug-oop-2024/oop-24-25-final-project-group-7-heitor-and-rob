from autoop.core.ml.model.model import Model
import numpy as np
import pandas as pd
from typing import Any
from copy import deepcopy


class KNearestNeighbors(Model):
    def __init__(self, k: int = 3, name: str = "K-Nearest Neighbors", type: str = "classification",) -> None:
        super().__init__(name=name, type=type)
        self.k = k
        self.observations = None
        self.ground_truth = None
        self._parameters = {}

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
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
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> Any:
        distances = np.linalg.norm(self.observations - observation, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.ground_truth[k_indices]
        most_common = pd.Series(k_nearest_labels).value_counts()
        return most_common.idxmax()
