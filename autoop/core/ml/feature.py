from typing import Literal, List, Any
from copy import deepcopy
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature():
    def __init__(self, name: str, type: Literal["numerical", "categorical"]) -> None:
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if len(name) == 0:
            raise ValueError("Name must be a non empty string")
        if isinstance(name, str):
            self._name = name

    @property
    def type(self) -> Literal["numerical", "categorical"]:
        return self._type

    @type.setter
    def type(self, type: Literal["numerical", "categorical"]) -> None:
        if type not in {"numerical", "categorical"}:
            raise ValueError(
                f"Type must be either 'numerical' or 'categorical'. Recieved {type}")
        self._type = type

    def __str__(self) -> str:
        return f"Feature(name={self._name}, type={self._type})"
