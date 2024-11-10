import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import LassoCV as SklearnLassoCV


class LassoCV(Model):
    def __init__(self, alphas: np.ndarray = np.logspace(-6, 6, 13)) -> None:
        """
        Initializes the lasso model with cross-validation for alpha selection
        :param alphas : Array of alpha values to be tested in cross-validation
        """
        super().__init__(type="regression")
        self.parameters = {"alphas": alphas}
        self.model = SklearnLassoCV(alphas=alphas)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the lasso model to the provided values, selecting the best alpha via cross-validation
        :param X : Numpy array of data
        :param Y : Numpy array of the target values corresponding to each observation
        """
        self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Takes the observations and calculates predictions using the fitted model
        :param X : Numpy array of data
        :return : Numpy array of predicted target values corresponding to the input observations
        """
        return self.model.predict(X)

    def get_best_alpha(self) -> float:
        """
        Returns the best alpha selected by cross-validation
        :return : Best alpha value
        """
        return self.model.alpha_