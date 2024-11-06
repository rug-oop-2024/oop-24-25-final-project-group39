import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import Ridge


class RidgeModel(Model):
    def __init__(self, alpha=1.0) -> None:
        super().__init__(type="regression")
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Takes the numpy arrays and calculates the parameters.
        :param x : Numpy array of data minus the variable to predict.
        :param y : Numpy array of the variable to predict.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Takes the observations and calculates predictions using
        the parameters.
        :param x : Numpy array of the other variables.
        :return : Numpy array of the predicted values.
        """
        return self.model.predict(X)