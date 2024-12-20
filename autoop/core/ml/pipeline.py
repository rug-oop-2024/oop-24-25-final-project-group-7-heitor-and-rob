from typing import List, Dict, Any
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    A class used to represent a Machine Learning Pipeline.

    Attributes
    ----------
    metrics : List[Metric]
        A list of metrics to evaluate the model.
    dataset : Dataset
        The dataset to be used in the pipeline.
    model : Model
        The model to be trained and evaluated.
    input_features : List[Feature]
        A list of input features.
    target_feature : Feature
        The target feature.
    split : float
        The ratio to split the dataset into training and testing sets.
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8) -> None:
        """
        Parameters
        ----------
        metrics : List[Metric]
            A list of metrics to evaluate the model.
        dataset : Dataset
            The dataset to be used in the pipeline.
        model : Model
            The model to be trained and evaluated.
        input_features : List[Feature]
            A list of input features.
        target_feature : Feature
            The target feature.
        split : float, optional
            The ratio to split the dataset into training and testing sets
            (default is 0.8).
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical":
            if model.type != "classification":
                raise ValueError(
                    """Model type must be classification
                  for categorical target feature"""
                )
        if target_feature.type == "numerical":
            if model.type != "regression":
                raise ValueError(
                    "Model type must be regression "
                    "for numerical target feature"
                )

    def __str__(self) -> str:
        """
        Returns a string representation of the Pipeline object.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Returns the model used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Returns the artifacts generated during the pipeline
          execution to be saved.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Dict[str, Any]) -> None:
        """
        Registers an artifact with the given name.

        Parameters
        ----------
        name : str
            The name of the artifact.
        artifact : Dict[str, Any]
            The artifact to be registered.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the input and target features.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """
        Splits the data into training and testing sets.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))]
                         for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):]
                        for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compacts a list of vectors into a single numpy array.

        Parameters
        ----------
        vectors : List[np.array]
            A list of numpy arrays to be compacted.

        Returns
        -------
        np.array
            A single numpy array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model using the testing data.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append(
                (f"{metric.__class__.__name__}: {float(result)}"))
        self._predictions = predictions

    def execute(self) -> Dict[str, Any]:
        """
        Executes the pipeline.

        Returns
        -------
        dict
            A dictionary containing the metrics results and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
