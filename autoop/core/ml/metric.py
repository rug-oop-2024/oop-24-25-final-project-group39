from abc import ABC, abstractmethod
from typing import Any, Union, Callable
import numpy as np
from math import sqrt

METRICS = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "r_squared",
    "accuracy",
    "micro_average_precision",
    "macro_average_precision",
    "micro_average_recall",
    "macro_average_recall"
]


def get_metric(name: str) -> Union["MeanAbsoluteError", "MeanSquaredError",
                                   "RootMeanSquaredError",
                                   "MeanAbsolutePercentageError",
                                   "RSquared", "Accuracy",
                                   "MicroAveragePrecision",
                                   "MacroAveragePrecision",
                                   "MicroAverageRecall",
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
        case "mean_absolute_percentage_error":
            return MeanAbsolutePercentageError()
        case "r_squared":
            return RSquared()
        case "accuracy":
            return Accuracy()
        case "micro_average_precision":
            return MicroAveragePrecision()
        case "macro_average_precision":
            return MacroAveragePrecision()
        case "micro_average_recall":
            return MicroAverageRecall()
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


class MeanAbsolutePercentageError(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Absolute Percentage Error (MAPE)
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        small_num = 1e-10  # Small value to avoid division by zero
        percentage_error = np.abs((y_true - y_pred) / (y_true + small_num))
        return np.mean(percentage_error) * 100


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


class MicroAveragePrecision(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the micro-average precision score
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        true_positive = 0
        false_positive = 0
        different_labels = np.unique(y_true)

        for label in different_labels:
            true_positive += np.sum((y_true == label) & (y_pred == label))
            false_positive += np.sum((y_true != label) & (y_pred == label))

        if true_positive + false_positive == 0:
            return 0.0
        else:
            micro_precision = true_positive / (true_positive + false_positive)
            return micro_precision


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


class MicroAverageRecall(Metric):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the micro-average recall score
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        true_positive = 0
        false_negative = 0
        different_labels = np.unique(y_true)

        for label in different_labels:
            true_positive += np.sum((y_true == label) & (y_pred == label))
            false_negative += np.sum((y_true == label) & (y_pred != label))

        if true_positive + false_negative == 0:
            return 0.0
        else:
            micro_recall = true_positive / (true_positive + false_negative)
            return micro_recall


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
