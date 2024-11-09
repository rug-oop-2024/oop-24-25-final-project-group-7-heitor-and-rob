from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self.type = "regression"
        self.parameters = {}
        self.initialize_model()

    def initialize_model(self):
        """Initialize a Scikit-learn LinearRegression model."""
        self._model = LinearRegression(**self._hyperparameters)
