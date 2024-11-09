from abc import ABC, abstractmethod
from typing import Any, Union, Callable
import numpy as np
from math import sqrt

METRICS = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared",
    "accuracy",
    "macro_average_precision",
    "macro_average_recall"
]


def get_metric(name: str) -> Union["MeanAbsoluteError", "MeanSquaredError",
                                   "RootMeanSquaredError", "RSquared",
                                   "Accuracy", "MacroAveragePrecision",
                                   "MacroAverageRecall"]:
    """
    Factory function to retrieve a metric by its name
    Args:
        name (str): The name of the metric to retrieve
    Returns:
        Union[MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
              RSquared, Accuracy, MacroAveragePrecision, MacroAverageRecall]:
              The corresponding metric object
    Raises:
        ValueError: If the provided metric name is not in the accepted list
    """
    if name not in METRICS:
        raise ValueError(f"{name} is not an accepted metric")
    match name:
        case "mean_absolute_error":
            return MeanAbsoluteError()
        case "mean_squared_error":
            return MeanSquaredError()
        case "root_mean_squared_error":
            return RootMeanSquaredError()
        case "r_squared":
            return RSquared()
        case "accuracy":
            return Accuracy()
        case "macro_average_precision":
            return MacroAveragePrecision()
        case "macro_average_recall":
            return MacroAverageRecall()


class Metric(ABC):
    """
    Base class for all metrics.
    """
    @abstractmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluates the metric
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        pass

    def __call__(self) -> Callable[[Any], Any]:
        """
        Callable to invoke the evaluate method directly
        Returns:
            Callable[[Any], Any]: The evaluate method
        """
        return self.evaluate()


class MeanAbsoluteError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Absolute Error
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        return np.mean(np.abs(y_true - y_pred))


class MeanSquaredError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean squared Error
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Root Mean Squared Error
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        return sqrt(np.mean((y_true - y_pred) ** 2))


class RSquared(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the R-squared, coefficient of determination
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        y_mean = np.mean(y_true)
        total_sum_squares = np.sum((y_true - y_mean) ** 2)
        residual_sum_squares = np.sum((y_true - y_pred) ** 2)
        if total_sum_squares == 0:
            return 1.0
        return 1 - (residual_sum_squares / total_sum_squares)


class Accuracy(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the accuracy score
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        return np.mean(y_true == y_pred)


class MacroAveragePrecision(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the macro-average precision score
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        different_labels = np.unique(y_true)
        precision_per_class = []
        for label in different_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            false_positive = np.sum((y_true != label) & (y_pred == label))
            if true_positive + false_positive == 0:
                precision = 0
            else:
                precision = true_positive / (true_positive + false_positive)
            precision_per_class.append(precision)
        return np.mean(precision_per_class)


class MacroAverageRecall(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the macro-average recall score
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        different_labels = np.unique(y_true)
        recall_per_class = []
        for label in different_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            false_negative = np.sum((y_true == label) & (y_pred != label))
            if true_positive + false_negative == 0:
                precision = 0
            else:
                precision = true_positive / (true_positive + false_negative)
            recall_per_class.append(precision)
        return np.mean(recall_per_class)
