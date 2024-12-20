from abc import ABC
import numpy as np
from copy import deepcopy
from autoop.core.ml.artifact import Artifact


class Model(ABC, Artifact):
    """
    Abstract base class for machine learning models.

    Attributes:
        _model: The machine learning model instance.
        type: The type of the model.
        _hyperparameters: The hyperparameters for the model.
        parameters: The parameters of the model after fitting.
    """

    def __init__(self, name: str, type: str) -> None:
        """
        Initialize the model with given hyperparameters.

        Args:
            hyperparameters: Arbitrary keyword arguments for hyperparameters.
        """
        super().__init__(name=name, type=type)

    def initialize_model(self) -> None:
        """
        Initialize the machine learning model.
        """
        pass

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the provided data.

        Args:
            x: Input features as a numpy array.
            y: Target values as a numpy array.
        """
        self.model.fit(x, y)
        self.parameters = {
            "strict_parameters": deepcopy(self.model.get_params()),
            "hyperparameters": self._hyperparameters
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.

        Args:
            x: Input features as a numpy array.

        Returns:
            Predictions as a numpy array.
        """
        return self.model.predict(x)
