from autoop.core.ml.model.model import Model
import numpy as np
from typing import Any
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):
    def __init__(self, n_trees: int = 100, max_depth: int = None, min_samples_split: int = 2, name: str = "Random Forest", type: str = "classification",) -> None:
        super().__init__(name=name, type=type)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._hyperparameters = {
            "n_estimators": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
        self._model = None

    def initialize_model(self) -> None:
        self._model = RandomForestClassifier(**self._hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._model is None:
            raise ValueError(
                "Model has not been initialized. Call `initialize_model()` first.")
        self._model.fit(X, y)
        self.parameters = {
            "strict_parameters": deepcopy(self._model.get_params()),
            "hyperparameters": self._hyperparameters
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError(
                "Model has not been initialized. Call `initialize_model()` first.")
        return self._model.predict(X)
