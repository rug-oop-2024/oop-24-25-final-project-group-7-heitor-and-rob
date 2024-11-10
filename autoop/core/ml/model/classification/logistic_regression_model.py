import numpy as np
from autoop.core.ml.model.model import Model


class LogisticRegression(Model):
    """Logistic Regression classifier."""

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 **hyperparameters) -> None:
        """
        Initialize the Logistic Regression model.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            num_iterations (int): Number of iterations for training.
            **hyperparameters: Additional hyperparameters.
        """
        super().__init__(**hyperparameters)
        self.type = "classification"
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}
        self.weights = None
        self.bias = None

    def initialize_model(self) -> None:
        """Initialize model parameters."""
        self.weights = None
        self.bias = 0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of z.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of the input.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Logistic Regression model.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.parameters = {"weights": self.weights, "bias": self.bias}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Logistic Regression model.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return (probabilities >= 0.5).astype(int)
