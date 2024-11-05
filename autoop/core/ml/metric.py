from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if len(name) == 0:
        raise ValueError("Name of metric must be a non empty string")
    if name.lower() == 'accuracy':
        return Accuracy()
    if name.lower() == "meansquarederror":
        return MeanSquaredError()

class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    def __init__(self) -> None:
        super().__init__()
        self._ground_truth = None
        self._prediction = None

    @abstractmethod
    def evaluate(self, prediction: float, ground_truth: float) -> int:
        pass


# add here concrete implementations of the Metric class
class Accuracy(Metric):
    def __init__(self):
        super().__init__()

    def evaluate(self, prediction: float, ground_truth: float) -> int:
        return np.mean(ground_truth == prediction)
    
class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__()

    def evaluate(self, prediction: float, ground_truth: float) -> int:
        return np.mean((prediction - ground_truth) ** 2)
