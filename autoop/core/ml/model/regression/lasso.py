import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import Lasso as SklearnLasso


class Lasso(Model):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(type="regression")
        self.hyperparameters = {"alpha": alpha}
        self._parameters = {
            "coefficients": self.lasso.coef_,
            "intercept": self.lasso.intercept_}
        self.model = SklearnLasso(alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the lasso model to the provided values.
        :param X : Numpy array of data.
        :param Y : Numpy array of the target values
        corresponding to each observation.
        """
        self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given observations.
        :param X : Numpy array of data.
        :return : Numpy array of predicted target values correspond to
        the input observations.
        """
        return self.model.predict(X)