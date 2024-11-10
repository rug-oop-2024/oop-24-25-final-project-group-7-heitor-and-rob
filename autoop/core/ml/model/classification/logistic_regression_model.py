from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model.model import Model


class LogisticRegressionModel(Model):
    """
    Logistic Regression Model for classification tasks.

    Attributes:
        type (str): Type of the model, set to "classification".
        parameters (dict): Dictionary to store model parameters.
    """

    def __init__(self, **hyperparameters):
        """
        Initialize the LogisticRegressionModel with given hyperparameters.

        Args:
            hyperparameters (dict): Hyperparameters for the Logistic Regression model.
        """
        super().__init__(**hyperparameters)
        self.type = "classification"
        self.parameters = {}
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize a Scikit-learn LogisticRegression model.
        """
        self._model = LogisticRegression(**self._hyperparameters)
