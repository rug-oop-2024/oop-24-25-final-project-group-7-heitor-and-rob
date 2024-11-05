from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    def initialize_model(self):
        """Initialize a Scikit-learn LinearRegression model for multiple linear regression."""
        return LinearRegression(**self.hyperparameters)
