from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
from typing import Callable
from math import sqrt

METRICS = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "accuracy",
    "precision",
    "recall"
]

def get_metric(name: str) -> Union["MeanAbsoluteError", "MeanSquaredError", 
                                   "RootMeanSquaredError", "Accuracy", 
                                   "Precision", "Recall"]:
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name not in METRICS:
        raise ValueError(f"{name} is not an accepted metric")
    match name:
        case "mean_absolute_error":
            return MeanAbsoluteError()
        case "mean_squared_error":
            return MeanSquaredError()
        case "root_mean_squared_error":
            return RootMeanSquaredError()
        case "accuracy":
            return Accuracy()
        case "precision":
            return Precision()
        case "recall":
            return Recall()

class Metric(ABC):
    """
    Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    @abstractmethod
    def evaluate(y_true, y_pred):
        pass

    def __call__(self) -> Callable[[any], any]:
        return self.evaluate() # Not sure about this
    

class MeanAbsoluteError(Metric):
    def evaluate(self, y_true, y_pred) -> float:
        return np.mean(np.abs(y_true - y_pred))

class MeanSquaredError(Metric):
    def evaluate(self, y_true, y_pred) -> float:
        return np.mean((y_true - y_pred) ** 2)

class RootMeanSquaredError(Metric):
    def evaluate(self, y_true, y_pred) -> float:
        # Maybe change this later to the next code if it works:
        # return sqrt(MeanSquaredError.evaluate(y_true-y_pred))
        return sqrt(np.mean((y_true - y_pred) ** 2))

class Accuracy(Metric):
    def evaluate(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

class Precision(Metric):
    def evaluate(self, y_true, y_pred):
        pass

class Recall(Metric):
    def evaluate(self, y_true, y_pred):
        pass