import numpy as np

from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(Model):
    """Logistic Regression model for classification using scikit-learn"""
    def __init__(self, penalty: str = 'l2',
                 C: float = 1.0, max_iter: int = 100) -> None:
        """
        Initializes the logistic regression model
        Args:
            penalty (str): Regularization technique, default is 'l2'
            C (float): Inverse of regularization strength
            max_iter (int): Maximum number of iterations
            for the model to converge
        Returns:
            None
        """
        super().__init__(type="classification",
                         parameters={"penalty": penalty,
                                     "C": C,
                                     "max_iter": max_iter})
        self.model = LogisticRegression(penalty=penalty,
                                        C=C,
                                        max_iter=max_iter)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the logistic regression model to the training data.
        :param observations: Numpy array where each row is a data point.
        :param ground_truth: Numpy array with the corresponding labels.
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the label for each observation in the input.
        :param observations: Numpy array where
        each row is a data point to predict.
        :return: Numpy array of predicted labels for each input example.
        """
        return self.model.predict(observations)
