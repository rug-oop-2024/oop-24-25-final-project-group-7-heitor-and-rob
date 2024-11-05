from sklearn.linear_model import LinearRegression
from .. import Model


class LinearRegressionModel(Model):
    def initialize_model(self):
        return LinearRegression(**self.hyperparameters)
