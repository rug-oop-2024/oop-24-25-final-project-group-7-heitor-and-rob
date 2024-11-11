from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy


class DecisionTree(Model):
    """
    Decision Tree model for classification tasks.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        name (str): Name of the model.
        type (str): Type of the model.
        _model (DecisionTreeClassifier): The DecisionTreeClassifier instance.
    """

    def __init__(self, max_depth: int = None, min_samples_split: int = 2,
                 name: str = "Decision Tree", type: str = "classification") -> None:
        """
        Initializes the DecisionTree model with given hyperparameters.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            name (str): Name of the model.
            type (str): Type of the model.
        """
        super().__init__(name=name, type=type)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._parameters = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
        self._model = DecisionTreeClassifier(**self._parameters)

    def fit(self, X, y) -> None:
        """
        Fits the DecisionTree model to the provided data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.

        Raises:
            ValueError: If the model has not been initialized.
        """
        # Ensure X and y are NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if self._model is None:
            raise ValueError("Model has not been initialized.")

        self._model.fit(X, y)
        self.parameters = {"strict_parameters": deepcopy(
            self._model.get_params())}

    def predict(self, X) -> np.ndarray:
        """
        Predicts the target values for the provided data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted target values.

        Raises:
            ValueError: If the model has not been initialized.
        """
        # Ensure X is a NumPy array
        X = np.asarray(X)

        if self._model is None:
            raise ValueError("Model has not been initialized.")

        return self._model.predict(X)
