
import os
import pickle
import numpy as np
from copy import deepcopy
from typing import Literal
from abc import ABC, abstractmethod

from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """Abstract base class for machine learning models"""
    def __init__(self, type: Literal["regression", "classification"],
                 parameters: dict = {}) -> None:
        """
        Initializes the model
        Args:
            type (Literal["regression", "classification"]): The type of model,
            either "regression" or "classification"
            parameters (dict)): A dictionary of parameters
        Returns:
            None
        """
        self._parameters = parameters
        self._type = type

    @property
    def parameters(self) -> dict:
        """
        Getter function for the private _parameters attribute
        Returns:
            dict: A deep copy of the parameters
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, new_parameters: dict) -> None:
        """
        Sets new model parameters
        Args:
            new_parameters (dict): A dictionary of new parameters
        Returns:
            None
        """
        self._parameters = new_parameters

    @property
    def type(self) -> Literal["regression", "classification"]:
        """
        Getter function for the model type
        Returns:
            Literal["regression", "classification"]: The type of the model
        """
        return self._type

    @type.setter
    def type(self, new_type: Literal["regression", "classification"]) -> None:
        """
        Sets a new model type
        Args:
            new_type (Literal["regression", "classification"]): The new type,
            either regression or classification to assign to the model
        Returns:
            None
        """
        self._type = new_type

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fits the model to the training data.
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

    def to_artifact(self, name: str) -> Artifact:
        """
        Makes the model class into a artifact
        Args:
            name (str): The name of the model
        Returns:
            Artifact: The data of the model stored in an artifact
        """
        return Artifact(
            name=name,
            asset_path=os.path.abspath(__file__),
            meta_data={},
            tags=[],
            data=pickle.dumps(self.parameters),
            type=self.type,
            version="1.0.0"
        )
