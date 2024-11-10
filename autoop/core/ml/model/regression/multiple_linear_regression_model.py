from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """
    A class used to represent a Multiple Linear Regression model.

    Attributes
    ----------
    type : str
        The type of the model, which is 'regression'.
    parameters : dict
        A dictionary to store the model parameters.
    """

    def __init__(self, **hyperparameters):
        """
        Initialize the MultipleLinearRegression model with given hyperparameters.

        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters for the LinearRegression model.
        """
        super().__init__(**hyperparameters)
        self.type = "regression"
        self.parameters = {}
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize a Scikit-learn LinearRegression model for multiple linear regression.

        Returns
        -------
        LinearRegression
            An instance of Scikit-learn's LinearRegression model.
        """
        return LinearRegression(**self._hyperparameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model by solving for the weight vector w using the closed-form solution for multiple linear regression.

        Parameters
        ----------
        observations : np.ndarray
            A 2D array of input data.
        ground_truth : np.ndarray
            A 1D array of target values.

        Raises
        ------
        ValueError
            If the dimensions of observations and ground_truth do not match,
            if observations is not a 2D array, or if input arrays are empty.
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

        self.parameters['weights'] = w_wave_star[:-1]
        self.parameters['biases'] = w_wave_star[-1]

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Make predictions for new input data.

        Parameters
        ----------
        observation : np.ndarray
            A 1D array of input data.

        Returns
        -------
        np.ndarray
            The predicted values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if 'weights' not in self.parameters or 'biases' not in self.parameters:
            raise ValueError("Model has not been fitted yet.")
        y_hat = observation.dot(self.parameters['weights'])
        return y_hat + self.parameters['biases']
