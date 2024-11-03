
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
# Don't know what for Artifact can be used
import numpy as np
from copy import deepcopy
from typing import Literal  # automatically checkes values


class Model(ABC):
    _parameters: dict
    _hyperparameters: dict
    _type: Literal["regression", "classification"]

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @property
    def hyperparameters(self) -> dict:
        return deepcopy(self._hyperparameters)

    @property
    def type(self) -> Literal["regression", "classification"]:
        return self._type

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """"Fits the model to the training data.
        :Param X : The input data for training.
        :Param Y : The target labels for training.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predicitions.
        :Param X : The input data for prediction.
        :Returns : Predicted values based on the input data.
        """
        pass
