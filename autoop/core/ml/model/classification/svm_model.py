from sklearn.svm import SVC
from .. import Model


class SVMClassifier(Model):
    def initialize_model(self):
        return SVC(**self.hyperparameters)
