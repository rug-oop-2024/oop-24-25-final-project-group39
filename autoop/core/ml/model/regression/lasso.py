import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import Lasso as SklearnLasso


class Lasso(Model):
    """Lasso regression model for regression tasks using scikit-learn"""
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes the lasso model
        Returns:
            None
        """
        super().__init__(type="regression", parameters={"alpha": alpha})
        self.model = SklearnLasso(alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the lasso model to the provided values
        :param X : Numpy array of data
        :param Y : Numpy array of the target values
        corresponding to each observation
        """
        self.model.fit(X, Y)
        self.parameters = {"coefficients": self.model.coef_,
                           "intercept": self.model.intercept_,
                           "alpha": self.model.alpha}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Takes the observations and calculates predictions using
        the parameters
        :param X : Numpy array of data
        :return : Numpy array of predicted target values correspond to
        the input observations
        """
        return self.model.predict(X)