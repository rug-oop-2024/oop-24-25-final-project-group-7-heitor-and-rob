from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model.model import Model


class LogisticRegressionModel(Model):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self.type = "classification"
        self.parameters = {}
        self.initialize_model()

    def initialize_model(self):
        """Initialize a Scikit-learn LogisticRegression model."""
        self._model = LogisticRegression(**self._hyperparameters)
