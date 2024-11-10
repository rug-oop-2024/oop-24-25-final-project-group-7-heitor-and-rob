import numpy as np
from autoop.core.ml.model.model import Model


class LogisticRegression(Model):
    def __init__(self, learning_rate=0.01, num_iterations=1000, **hyperparameters):
        super().__init__(**hyperparameters)
        self.type = "classification"
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.parameters = {}
        self.weights = None
        self.bias = None

    def initialize_model(self):
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
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

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return (probabilities >= 0.5).astype(int)
