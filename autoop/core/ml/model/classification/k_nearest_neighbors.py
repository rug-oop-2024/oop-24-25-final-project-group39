import numpy as np

from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Model):
    """K-Nearest Neighbors model for classification using scikit-learn"""
    def __init__(self, n_neighbors: int = 3) -> None:
        """
        Initializes the knn model
        Args:
            n_neighbors (int): The number of neighbors to use for prediction,
            defaults to 3
        Returns:
            None
        """
        super().__init__(type="classification",
                         parameters={"n_neighbors": n_neighbors})
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the model to the training data
        Args:
            observations (np.ndarray): Training data where
            each row is a data point
            ground_truth (np.ndarray): Target labels for each data
            point in the training set
        Returns:
            None
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts labels for each observation in the input
        Args:
            observations (np.ndarray): Input data features for prediction
        Returns:
            np.ndarray: Predicted labels for each input observation
        """
        return self.model.predict(observations)
