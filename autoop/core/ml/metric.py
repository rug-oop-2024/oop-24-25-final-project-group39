from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from typing import Callable

METRICS = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "accuracy",
    "precision",
    "recall"
]

def get_metric(name: str):
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
    def _calculate(self) -> any:
        pass

    def __call__(self) -> Callable[[any], any]:
        return self._calculate()
    

# add here concrete implementations of the Metric class
class MeanAbsoluteError(Metric):
    pass

class MeanSquaredError(Metric):
    pass

class RootMeanSquaredError(Metric):
    pass

class Accuracy(Metric):
    pass

class Precision(Metric):
    pass

class Recall(Metric):
    pass