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
        self._model = None

    def initialize_model(self) -> None:
        """
        Initialize the Lasso model with the specified hyperparameters.
        """
        self._model = SklearnLasso(**self._parameters)

    @property
    def parameters(self) -> dict:
        """
        Get the hyperparameters of the model.

        :return: A dictionary of hyperparameters.
        """
        return deepcopy(self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Lasso model to the given data.

        :param observations: Training data.
        :param ground_truth: Target values.
        :raises ValueError: If the model has not been initialized.
        """
        if self._model is None:
            self.intialize_model()

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
