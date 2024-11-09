import numpy as np

from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Model):
    """K-Nearest Neighbors model for classification using scikit-learn"""
    def __init__(self, n_neighbors: int = 3) -> None:
        """
        Initializes the knn model
        Returns:
            None
        """
        super().__init__(type="classification",
                         parameters={"n_neighbors": n_neighbors})
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
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
