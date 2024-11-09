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
    @abstractmethod
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        pass


# add here concrete implementations of the Metric class
class Accuracy(Metric):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean(ground_truth == prediction)
    
class MeanSquaredError(Metric):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean((prediction - ground_truth) ** 2)
    
class Precision(Metric):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        classes = np.unique(ground_truth)
        precisions = []
        for category in classes:
            true_positives = np.sum((prediction == category) & (ground_truth == category))
            predicted_positives = np.sum(prediction == category)
            if predicted_positives == 0:
                precisions.append(0)
            else:
                precisions.append(true_positives / predicted_positives)
        return precisions/len(classes)
            
    



