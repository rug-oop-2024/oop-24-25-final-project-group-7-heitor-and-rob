"""
This module provides a factory function to get machine learning models
by task type and model name.
"""

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression_model import (
    LinearRegression,
)
from autoop.core.ml.model.regression.multiple_linear_regression_model import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_model import Lasso
from autoop.core.ml.model.classification.KNN_model import KNearestNeighbors
from autoop.core.ml.model.classification.logistic_regression_model import (
    LogisticRegression,
)
from autoop.core.ml.model.classification.random_forest_model import (
    RandomForest,
)
from typing import Type


REGRESSION_MODELS = [
    "linear_regression",
    "multiple_linear_regression",
    "lasso"
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "KNN"
]


def get_model(name: str) -> Type[Model]:
    """
    Factory function to get a model by task type and model name.

    Parameters:
    task_type (str): The type of task ('classification' or 'regression').
    model_name (str): The name of the model.
    **hyperparameters: Additional hyperparameters for the model.

    Returns:
    Model: An instance of the requested model.

    Raises:
    ValueError: If the task type or model name is not supported.
    """
    model_map = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForest,
        "KNN": KNearestNeighbors,
        "multiple_linear_regression": MultipleLinearRegression,
        "linear_regression": LinearRegression,
        "lasso": Lasso
    }

    if name not in model_map:
        raise ValueError(
            f"Unsupported task type: {name}. Supported task types are: "
            f"{list(model_map.keys())}"
        )

    model = model_map[name]
    task_type = "classification" if name in CLASSIFICATION_MODELS else "regression"
    return model(name=name, type=task_type)
