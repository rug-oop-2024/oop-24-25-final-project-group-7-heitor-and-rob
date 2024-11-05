from sklearn.svm import SVC
from ml.model import Model


class SVMClassifier(Model):
    def initialize_model(self):
        return SVC(**self.hyperparameters)
