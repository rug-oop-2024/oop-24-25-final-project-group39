import numpy as np

from autoop.core.ml.model import Model
from sklearn.svm import LinearSVC


class LinearSupportVectorClassifier(Model):
    """Linear Support Vector Classifier using scikit-learn's LinearSVC"""
    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        """
        Initializes the Linear Support Vector Classifier
        Args:
            C (float): Regularization parameter, defaults to 1.0
            max_iter (int): The maximum number of iterations, defaults to 1000
        Returns:
            None
        """
        super().__init__(type="classification",
                         parameters={"C": C, "max_iter": max_iter})
        self.model = LinearSVC(C=C, max_iter=max_iter)

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
            np.ndarray: Predicted labels for each input observation.
        """
        return self.model.predict(observations)
