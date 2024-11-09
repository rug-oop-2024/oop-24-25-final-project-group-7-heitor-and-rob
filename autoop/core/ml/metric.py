from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "root_mean_squared_error",
    "r_squared",
    "precision",
    "recall",
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
    if name.lower() == "root_mean_squared_error":
        return RootMeanSquaredError()
    if name.lower() == "r_squared":
        return Rsquared()
    if name.lower() == "precision":
        return Precision()
    if name.lower() == "recall":
        return Recall()

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
    
class RootMeanSquaredError(MeanSquaredError):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return  np.sqrt(super().evaluate(prediction, ground_truth))
    
class Rsquared(Metric):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        y_mean = np.mean(ground_truth)
        total = np.sum((ground_truth - y_mean) ** 2)
        residual = np.sum((ground_truth - prediction) ** 2)
        return 1 - (residual / total)
    
class Precision(Metric):
    #classification
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
    
class Recall(Metric):
    #classification
    def evaluate(self, prediction, ground_truth):
        classes = np.unique(ground_truth)
        recall_list = []
        for category in classes:
            true_positives = np.sum((prediction == category) & (ground_truth == category))
            false_negatives = np.sum((ground_truth == category) & (prediction != category))
            if true_positives + false_negatives == 0:
                recall_list.append(0)
            else:
                recall_list.append(true_positives/(true_positives+false_negatives))
        return recall_list/len(classes)





