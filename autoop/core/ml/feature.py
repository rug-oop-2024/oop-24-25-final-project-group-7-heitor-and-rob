
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type

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
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        if len(type) == 0:
            raise ValueError("Type must be a non empty string")
        if type.lower() not in ["numerical","categorical"]:
            raise ValueError(f"Type must be either numerical or categorical. Recieved {type}")
        self._type = type.lower()