from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from copy import deepcopy


class Lasso(Model):
    """
    A wrapper around the Lasso regression model from scikit-learn.
    This model is used for linear regression with L1 regularization.
    """

    def __init__(self, alpha: float = 1.0, name: str = "Lasso", type: str = "regression", **hyperparameters) -> None:
        super().__init__(name=name, type=type)
        self._hyperparameters = {
            "alpha": alpha,
            **hyperparameters
        }
        self._model = None

    def initialize_model(self) -> None:
        self._model = SklearnLasso(**self._hyperparameters)

    @property
    def hyperparameters(self) -> dict:
        return deepcopy(self._hyperparameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        if self._model is None:
            raise ValueError(
                "Model has not been initialized. Call `initialize_model()` first.")

        self._model.fit(observations, ground_truth)
        self.parameters = {
            "coef_": deepcopy(self._model.coef_),
            "intercept_": deepcopy(self._model.intercept_),
            "hyperparameters": self._hyperparameters
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if self._model is None or not hasattr(self._model, "coef_"):
            raise ValueError("Model has not been fitted yet.")
        return self._model.predict(observations)
