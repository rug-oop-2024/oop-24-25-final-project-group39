
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
# Don't know what for Artifact can be used
import numpy as np
from copy import deepcopy
from typing import Literal  # automatically checkes values


class Model(ABC):
    def __init__(self, type):
        self._parameters: dict
        self._hyperparameters: dict
        self._type: Literal["regression", "classification"] = type

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)
    
    @parameters.setter
    def parameters(self, new_parameters: dict) -> None:
        self._parameters = new_parameters

    @property
    def hyperparameters(self) -> dict:
        return deepcopy(self._hyperparameters)
    
    @hyperparameters.setter
    def hyperparameters(self, new_hyperparameters: dict) -> None:
        self._hyperparameters = new_hyperparameters

    @property
    def type(self) -> Literal["regression", "classification"]:
        return self._type
    
    @type.setter
    def type(self, new_type: Literal["regression", "classification"]) -> None:
        self._type = new_type

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

    def save_model(self):
        pass

    def load_model(self):
        pass