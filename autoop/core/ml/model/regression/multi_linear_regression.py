import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    """Multiple linear regression model for
    regression tasks using scikit-learn"""
    def __init__(self) -> None:
        """
        Initializes the multiple linear regression model
        Returns:
            None
        """
        super().__init__(type="regression", parameters={})
        self.model = LinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
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
        self.model.fit(x, y)
        self.parameters = {"coef_": self.model.coef_,
                           "intercept_": self.model.intercept_}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels for each observation in the input
        Args:
            observations (np.ndarray): Input data features for prediction
        Returns:
            np.ndarray: Predicted labels for each input observation
        """
        return self.model.predict(x)
