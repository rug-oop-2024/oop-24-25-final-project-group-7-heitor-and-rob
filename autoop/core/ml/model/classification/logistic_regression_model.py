import numpy as np
from autoop.core.ml.model.model import Model


class LogisticRegression(Model):
    """Simple Logistic Regression classifier."""

    def __init__(self, name: str = "Logistic Regression", type: str = "classification", learning_rate: float = 0.01, num_iterations: int = 1000):
        """
        Initialize the Logistic Regression model with given hyperparameters.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for gradient descent. Defaults to 0.01.
        num_iterations : int, optional
            The number of iterations for gradient descent. Defaults to 1000.
        """
        super().__init__(name=name, type=type)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function element-wise to the input array.

        Parameters
        ----------
        z : np.ndarray
            The input array to which the sigmoid function is applied.

        Returns
        -------
        np.ndarray
            The output array with the sigmoid function applied.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the given data.

        Parameters
        ----------
        X : np.ndarray
            A 2D array of input data.
        y : np.ndarray
            A 1D array of target values.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Input features as a numpy array.

        Returns:
            Predictions as a numpy array.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return (probabilities >= 0.5).astype(int)
