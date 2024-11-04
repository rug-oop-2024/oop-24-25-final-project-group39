from abc import ABC, abstractmethod
from typing import Any, Union, Callable
import numpy as np
from math import sqrt

METRICS = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared"
    "accuracy",
    "macro_average_precision",
    "macro_average_recall"
]


def get_metric(name: str) -> Union["MeanAbsoluteError", "MeanSquaredError",
                                   "RootMeanSquaredError", "Accuracy",
                                   "MacroAveragePrecision", "MacroAverageRecall"]:
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
            return MacroAveragePrecision()
        case "recall":
            return MacroAverageRecall()


class Metric(ABC):
    """
    Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction
    # as input and return a real number
    @abstractmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __call__(self) -> Callable[[Any], Any]:
        return self.evaluate()  # Not sure about this


class MeanAbsoluteError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class MeanSquaredError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return sqrt(np.mean((y_true - y_pred) ** 2))
    
class RSquared(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_mean = np.mean(y_true)
        total_sum_squares = np.sum((y_true - y_mean) ** 2)
        residual_sum_squares = np.sum((y_true - y_pred) ** 2)
        if total_sum_squares == 0: 
            return 1.0
        return 1 - (residual_sum_squares / total_sum_squares)

class Accuracy(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


