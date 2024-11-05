from sklearn.linear_model import LogisticRegression
from ml.model import Model


class LogisticRegressionModel(Model):
    def initialize_model(self):
        return LogisticRegression(**self.hyperparameters)
