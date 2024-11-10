import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import LassoCV as SklearnLassoCV


class LassoCV(Model):
    def __init__(self) -> None:
        """
        Initializes the lasso model with cross-validation for alpha selection
        :param alphas : Array of alpha values to be tested in cross-validation
        """
        super().__init__(type="regression", parameters={})
        self.model = SklearnLassoCV()

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
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
        self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for each observation in the input
        Args:
            observations (np.ndarray): Input data features for prediction
        Returns:
            np.ndarray: Predicted labels for each input observation
        """
        return self.model.predict(X)

    def get_best_alpha(self) -> float:
        """
        Returns the best alpha selected by cross-validation
        :return : Best alpha value
        """
        return self.model.alpha_
