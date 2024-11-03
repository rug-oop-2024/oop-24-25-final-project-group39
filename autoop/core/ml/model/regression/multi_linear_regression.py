import numpy as np
from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    def __init__(self) -> None:
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Takes the numpy arrays and calculates the parameters.
        :param x : Numpy array of data minus the variable to predict.
        :param y : Numpy array of the variable to predict.
        """
        self.model.fit(x, y)
        self._params = {"coef_": self.model.coef_, "intercept_": self.model.intercept_}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Takes the observations and calculates predictions using
        the parameters.
        :param x : Numpy array of the other variables.
        :return : Numpy array of the predicted values.
        """
        return self.model.predict(x)
