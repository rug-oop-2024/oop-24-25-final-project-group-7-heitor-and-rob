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
        return Accuracy
    if name.lower() == "meansquarederror":
        return MeanSquaredError

class Metric(ABC):
    """Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    def __init__(self, ground_truth: float, prediction: float) -> None:
        super().__init__()
        self._ground_truth = ground_truth
        self._prediction = prediction

    @abstractmethod
    def calculate(self) -> int:
        pass

    def __call__(self) -> int:
        return self.calculate()

# add here concrete implementations of the Metric class
class Accuracy(Metric):
    def __init__(self, ground_truth, prediction):
        super().__init__(ground_truth, prediction)

    def calculate(self):
        return np.mean(self._ground_truth == self._prediction)
    
class MeanSquaredError(Metric):
    def __init__(self, ground_truth, prediction):
        super().__init__(ground_truth, prediction)

    def calculate(self):
        return np.mean((self._prediction - self._ground_truth) ** 2)
    