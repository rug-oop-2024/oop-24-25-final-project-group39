from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):
    def __init__(self, type: Literal["regression", "classification"],
                 parameters: dict = {}):
        self._parameters = parameters
        self._type = type

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, new_parameters: dict) -> None:
        self._parameters = new_parameters

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
