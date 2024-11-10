import numpy as np
from autoop.core.ml.model.model import Model


class LinearRegression(Model):
    """
    Custom implementation of Linear Regression.

    Attributes:
        type (str): The type of the model, which is 'regression'.
        parameters (dict): The parameters of the model including weights and bias.
    """

    def __init__(
        self,
        name: str = "Linear Regression",
        type: str = "regression",
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        use_gradient_descent: bool = True,
        **hyperparameters
    ) -> None:
        """
        Initialize the LinearRegressionModel with given hyperparameters.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            num_iterations (int): The number of iterations for gradient descent.
            use_gradient_descent (bool): Whether to use gradient descent or the
                                         closed-form solution.
            **hyperparameters: Arbitrary keyword arguments for model
                               hyperparameters.
        """
        super().__init__(name=name, type=type)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.use_gradient_descent = use_gradient_descent
        self.parameters = {}
        self.weights = None
        self.bias = None

    def initialize_model(self) -> None:
        """Initialize the weights and bias."""
        self.weights = None
        self.bias = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model using gradient descent or the normal
        equation.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        X = np.c_[np.ones(X.shape[0]), X]
        num_samples, num_features = X.shape

        if self.use_gradient_descent:
            self.weights = np.zeros(num_features)

            for i in range(self.num_iterations):
                predictions = np.dot(X, self.weights)
                error = predictions - y
                gradient = (1 / num_samples) * np.dot(X.T, error)
                self.weights -= self.learning_rate * gradient
        else:
            self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        self.bias = self.weights[0]  # First element is the bias term
        self.weights = self.weights[1:]
        self.parameters = {"weights": self.weights, "bias": self.bias}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous values using the trained linear regression model.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
