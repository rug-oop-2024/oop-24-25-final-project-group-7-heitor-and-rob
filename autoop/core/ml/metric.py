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
    
class Macro_Recall(Metric):
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        num_classes = len(np.unique(ground_truth))
        recall = 0
        for class_ in list(ground_truth.unique()):
            # all classes except current are considered negative
            temp_true = [1 if p == class_ else 0 for p in ground_truth]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        
        # compute true positive for current class
            tp = true_positive(temp_true, temp_pred)



                    # compute false negative for current class
            fn = false_negative(temp_true, temp_pred)
            
            
            # compute recall for current class
            temp_recall = tp / (tp + fn + 1e-6)
            
            # keep adding recall for all classes
            recall += temp_recall
            
        # calculate and return average recall over all classes
        recall /= num_classes
        
        return recall


