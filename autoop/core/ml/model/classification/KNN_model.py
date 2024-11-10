from autoop.core.ml.model.model import Model
import numpy as np
import pandas as pd
from copy import deepcopy


class KNearestNeighbors(Model):
    """
    K-Nearest Neighbors classifier.
    """

    def __init__(self, k: int = 3, name: str = "K-Nearest Neighbors",
                 type: str = "classification") -> None:
        """
        Initialize the K-Nearest Neighbors model with given hyperparameters.

        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider for classification.
            Defaults to 3.
        name : str, optional
            The name of the model. Defaults to "K-Nearest Neighbors".
        type : str, optional
            The type of the model. Defaults to "classification".

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

        Returns
        -------
        dict
            A dictionary with the model parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """
        Set the model parameters.

        Parameters
        ----------
        value : dict
            A dictionary containing the parameters to set for the model.
        """
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the K-Nearest Neighbors model to the given data.

        Parameters
        ----------
        observations : np.ndarray
            A 2D array of input data.
        ground_truth : np.ndarray
            A 1D array of target values.

        Raises
        ------
        ValueError
            If the number of observations does not match the number of ground truth labels.
        """
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                "The number of observations must match the number of ground truth labels."
            )

        if len(ground_truth.shape) > 1:
            ground_truth = ground_truth.flatten()

        self.observations = observations
        self.ground_truth = ground_truth
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the given observations using the K-Nearest Neighbors
        algorithm.

        Parameters
        ----------
        observations : np.ndarray
            The observations to predict the labels of.

        Returns
        -------
        np.ndarray
            A 1D array of the predicted labels. The length of the array is equal to
            the number of observations.
        """
        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)

        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> np.ndarray:
        """
        Predicts the label of a single observation using the K-Nearest Neighbors
        algorithm.

        Parameters
        ----------
        observation : np.ndarray
            The observation to predict the label of.

        Returns
        -------
        np.ndarray
            The predicted label of the observation.
        """
        distances = np.linalg.norm(self.observations - observation, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.ground_truth[k_indices]
        most_common = pd.Series(k_nearest_labels).value_counts()
        return most_common.idxmax()
