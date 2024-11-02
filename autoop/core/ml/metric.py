from abc import ABC, abstractmethod
from typing import Any, Union, Callable
import numpy as np
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
    # remember: metrics take ground truth and prediction
    # as input and return a real number
    @abstractmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __call__(self) -> Callable[[any], any]:
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


class Accuracy(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class Precision(Metric): # Precision = TruePositive/ (Truepositive + FalsePositive)
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 0))
        return true_positive / (true_positive + false_positive)


class Recall(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_negative = np.sum((y_true == 0) & (y_pred == 1))
        return true_positive / (true_positive + false_negative)
