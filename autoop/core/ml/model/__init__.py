
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression_model import LinearRegression
from autoop.core.ml.model.regression.multiple_linear_regression_model import MultipleLinearRegression
from autoop.core.ml.model.classification.svm_model import SVMClassifier
from autoop.core.ml.model.classification.logistic_regression_model import LogisticRegression

REGRESSION_MODELS = [
    "linear_regression",
    "multiple_linear_regression",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "svm",
]  # add your models as str here


def get_model(task_type: str, model_name: str, **hyperparameters) -> Model:
    """Factory function to get a model by task type and model name."""
    model_map = {
        "classification": {
            "logistic_regression": LogisticRegression,
            "svm": SVMClassifier,
        },
        "regression": {
            "multiple_linear_regression": MultipleLinearRegression,
            "linear_regression": LinearRegression
        },
    }

    if task_type not in model_map:
        raise ValueError(
            f"Unsupported task type '{task_type}'. Choose either 'classification' or 'regression'.")

    task_models = model_map[task_type]
    if model_name not in task_models:
        available_models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
        raise ValueError(
            f"{task_type.capitalize()} model '{model_name}' is not supported. Available models: {available_models}")

    model = task_models[model_name]
    return Model(task_type=task_type, model=model, **hyperparameters)
