from autoop.core.ml.model.model import Model
import numpy as np
# from typing import Any  # Unused import removed
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):
    """
    RandomForest model for classification tasks.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to 
                                 split an internal node.
        name (str): Name of the model.
        type (str): Type of the model.
        _hyperparameters (dict): Hyperparameters for the 
                                 RandomForestClassifier.
        _model (RandomForestClassifier): The RandomForestClassifier 
                                         instance.
    """

    def __init__(self, n_trees: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, name: str = "Random Forest", 
                 type: str = "classification") -> None:
        """
        Initializes the RandomForest model with given hyperparameters.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to 
                                     split an internal node.
            name (str): Name of the model.
            type (str): Type of the model.
        """
        super().__init__(name=name, type=type)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._hyperparameters = {
            "n_estimators": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
        self._model = None

    def initialize_model(self) -> None:
        """
        Initializes the RandomForestClassifier with the specified 
        hyperparameters.
        """
        self._model = RandomForestClassifier(**self._hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the RandomForest model to the provided data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self._model is None:
            raise ValueError(
                "Model has not been initialized. Call `initialize_model()` first."
            )
        self._model.fit(X, y)
        self.parameters = {
            "strict_parameters": deepcopy(self._model.get_params()),
            "hyperparameters": self._hyperparameters
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the provided data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted target values.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self._model is None:
            raise ValueError(
                "Model has not been initialized. Call `initialize_model()` first."
            )
        return self._model.predict(X)
