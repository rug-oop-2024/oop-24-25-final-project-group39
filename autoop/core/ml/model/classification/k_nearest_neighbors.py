import numpy as np

from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Model):
    def __init__(self) -> None:
        super().__init__(type="classification")
        self.model = KNeighborsClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        """
        Fits the data by storing the observations
        and their corresponding labels.
        :param observations : Numpy array where each row is a data point.
        :param ground_truth : Numpy array with the corresponding labels.
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the label for each observation in the input.
        :param observations : Numpy array where
          each row is a data point to predict.
        :return : Numpy array of predicted labels for each input observation.
        """
        return self.model.predict(observations)