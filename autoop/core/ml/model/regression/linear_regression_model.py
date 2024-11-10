from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    """
    Linear Regression Model class that uses Scikit-learn's LinearRegression.
    
    Attributes:
        type (str): The type of the model, which is 'regression'.
        parameters (dict): The parameters of the model.
    """

    def __init__(self, **hyperparameters):
        """
        Initialize the LinearRegressionModel with given hyperparameters.

        Args:
            **hyperparameters: Arbitrary keyword arguments for model hyperparameters.
        """
        super().__init__(**hyperparameters)
        self.type = "regression"
        self.parameters = {}
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize a Scikit-learn LinearRegression model.
        """
        self._model = LinearRegression(**self._hyperparameters)
