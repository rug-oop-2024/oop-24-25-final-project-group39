from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """Machine learning pipeline for data processing,
    training, and evaluation"""
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initializes the pipeline with the parameters
        Args:
            metrics (List[Metric]): A list of metrics to evaluate the model
            dataset (Dataset): The dataset to be used in the pipeline
            model (Model): The model to be trained and evaluated
            input_features (List[Feature]): List of input features
            target_feature (Feature): The target feature
            split (float): The fraction of data to use for training,
            default is 0.8
        Returns:
            None
        Raises:
            ValueError: If the model type does not match
            the target feature type
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

        if target_feature.type == "categorical" and \
                model.type != "classification":
            raise ValueError(
                "Model type must be classification for "
                "categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """
        Returns a string representation of the pipeline
        Returns:
            str: The string representation of the pipeline
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
        Returns the model in the pipeline
        Returns:
            Model: The model
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during
        the pipeline execution to be saved
        Returns:
            List[Artifact]: List of artifact objects
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
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Registers an artifact to the pipeline
        Args:
            name (str): The name of the artifact
            artifact: The artifact object to register
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the freatures
        Returns:
            None
        """
        (target_feature_name, target_data, artifact) = \
            preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = \
            preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self) -> None:
        """
        Splits the data into training and testing sets
        Returns:
            None
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))]
                         for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):]
                        for vector in self._input_vectors]
        split_len = int(split * len(self._output_vector))
        self._train_y = self._output_vector[:split_len]
        self._test_y = self._output_vector[split_len:]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Combines the input vectors into a single 2D array
        Args:
            vectors (List[np.array]): A list of numpy arrays to
            be linked together
        Returns:
            np.array: The conjoint numpy array
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        Executes the entire pipeline, including preprocessing,
        training, and evaluation
        Returns:
            dict: A dictionary containing
            the evaluation metrics and predictions
            for testing and training sets
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()

        train_x = self._compact_vectors(self._train_X)
        test_x = self._compact_vectors(self._test_X)
        train_predictions = self._model.predict(train_x)
        train_metrics_results = []
        for metric in self._metrics:
            train_metrics_results.append((metric,
                                         metric.evaluate(train_predictions,
                                                         self._train_y)))

        test_predictions = self._model.predict(test_x)
        test_metrics_results = []
        for metric in self._metrics:
            test_metrics_results.append((metric,
                                         metric.evaluate(test_predictions,
                                                         self._test_y)))

        return {"train_predictions": train_predictions,
                "test_predictions": test_predictions,
                "train_metrics": train_metrics_results,
                "test_metrics": test_metrics_results}

    def to_artifact(self, name: str, version: str) -> "Artifact":
        """
        Serializes the model or object and converts it into an Artifact
        Args:
            name (str): The name to assign to the artifact
            version (str): The version identifier for the artifact
        Returns:
            Artifact: An Artifact object containing the serialized model/data
        """
        data = pickle.dumps(self)
        return Artifact(name=name, data=data,
                        asset_path=f"pipeline/{name}",
                        type="pipeline",
                        version=version)