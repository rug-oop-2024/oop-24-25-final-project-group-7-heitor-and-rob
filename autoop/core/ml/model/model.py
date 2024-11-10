
from abc import abstractmethod, ABC
import numpy as np
from copy import deepcopy


class Model(ABC):
    def __init__(self, **hyperparameters):
        self._model = None
        self.type = None
        self._hyperparameters = deepcopy(hyperparameters)
        self.parameters = None

    def initialize_model(self) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)
        self.parameters = {
            "strict_parameters": deepcopy(self.model.get_params()),
            "hyperparameters": self._hyperparameters
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)
