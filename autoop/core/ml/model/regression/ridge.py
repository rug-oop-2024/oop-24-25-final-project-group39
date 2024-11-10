import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import Ridge


class RidgeModel(Model):
    """Ridge regression model using scikit-learn for regression tasks"""
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes the ridge model
        Returns:
            None
        """
        super().__init__(type="regression", parameters={"alpha": alpha})
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for each observation in the input
        Args:
            observations (np.ndarray): Input data features for prediction
        Returns:
            np.ndarray: Predicted labels for each input observation
        """
        return self.model.predict(X)
