from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    def initialize_model(self):
        """Initialize a Scikit-learn LinearRegression model for multiple linear regression."""
        return LinearRegression(**self.hyperparameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the model by solving for the weight vector w.

        Using the closed-form solution for multiple linear regression.
        """
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError("Observations must match ground truth")
        if len(observations.shape) != 2:
            raise ValueError("Observations must be a 2D array.")
        if ground_truth.size == 0 or observations.size == 0:
            raise ValueError("Input arrays cannot be empty.")

        data_points = observations.shape[0]
        ones_column = np.ones((data_points, 1))
        x_wave = np.hstack([observations, ones_column])
        x_wave_transpose = x_wave.T
        w_wave = np.linalg.pinv(x_wave_transpose.dot(x_wave))
        w_wave_star = w_wave.dot(x_wave_transpose).dot(ground_truth)

        self._parameters['weights'] = w_wave_star[:-1]
        self._parameters['biases'] = w_wave_star[-1]

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Make predictions for new input data."""
        if 'weights' not in self._parameters or \
                'biases' not in self._parameters:
            raise ValueError("Model has not been fitted yet.")
        y_hat = observation.dot(self._parameters['weights'])
        return y_hat + self._parameters['biases']
