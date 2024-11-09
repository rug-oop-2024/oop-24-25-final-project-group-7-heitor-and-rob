from sklearn.svm import SVC
from autoop.core.ml.model.model import Model


class SVMClassifier(Model):
    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
        self.type = "classification"
        self.parameters = {}
        self.initialize_model()

    def initialize_model(self):
        """Initialize a Scikit-learn SVC model."""
        self._model = SVC(**self._hyperparameters)
