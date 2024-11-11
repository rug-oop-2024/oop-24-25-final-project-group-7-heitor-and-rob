import numpy as np
from autoop.core.ml.model.model import Model
from copy import deepcopy


class LinearRegression(Model):
    """
    A simple linear regression model using basic matrix operations.
    """

    def __init__(self,
                  name: str = "Simple Linear Regression",
                    type: str = "regression") -> None:
        """
        Initialize the Simple Linear Regression model.
        """
        super().__init__(name=name, type=type)
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model using the normal equation.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        y = y.flatten()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.weights + self.bias

    @property
    def parameters(self) -> dict:
        """
        Get the model parameters.

        Returns:
            dict: A dictionary containing the weights and bias.
        """
        return {
            "weights": deepcopy(self.weights),
            "bias": deepcopy(self.bias)
        }
