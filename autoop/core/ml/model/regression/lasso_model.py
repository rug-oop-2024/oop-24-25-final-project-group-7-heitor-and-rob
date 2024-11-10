from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from copy import deepcopy


class Lasso(Model):
    """
    A wrapper around the Lasso regression model from scikit-learn.
    This model is used for linear regression with L1 regularization.
    """

    def __init__(
            self, alpha: float = 1.0,
            name: str = "Lasso", type: str = "regression") -> None:
        """
        Initialize the Lasso model with given hyperparameters.

        :param alpha: Regularization strength.
        :param name: Name of the model.
        :param type: Type of the model.
        :param hyperparameters: Additional hyperparameters.
        """
        super().__init__(name=name, type=type)
        self._parameters = {
            "alpha": alpha
        }
        self._model = SklearnLasso(**self._parameters)

    @property
    def parameters(self) -> dict:
        """
        Get the hyperparameters of the model.

        :return: A dictionary of hyperparameters.
        """
        parameters = {
            **self._model.get_params(),
            "fitted_parameters": getattr(self._model, "coef_", None),
            "intercept": getattr(self._model, "intercept_", None),
        }
        return parameters

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """
        Set the hyperparameters of the model.

        :param value: A dictionary of hyperparameters.
        """
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Lasso model to the given data.

        :param observations: Training data.
        :param ground_truth: Target values.
        :raises ValueError: If the model has not been initialized.
        """

        self._model.fit(observations, ground_truth)
        self.parameters = {
            "coef_": deepcopy(self._model.coef_),
            "intercept_": deepcopy(self._model.intercept_)
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict using the Lasso model.

        :param observations: Samples.
        :return: Predicted values.
        :raises ValueError: If the model has not been fitted yet.
        """
        if self._model is None or not hasattr(self._model, "coef_"):
            raise ValueError("Model has not been fitted yet.")
        return self._model.predict(observations)
