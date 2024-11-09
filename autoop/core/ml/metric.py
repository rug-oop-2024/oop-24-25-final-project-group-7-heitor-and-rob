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
    
class RootMeanSquaredError(MeanSquaredError):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return  np.sqrt(super().evaluate(prediction, ground_truth))
    
class Rsquared(Metric):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray):
        y_minus = np.mean(ground_truth)
        return 1 - ((sum(ground_truth-prediction)**2)/sum(ground_truth-y_minus)**2)
    
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
            
    



