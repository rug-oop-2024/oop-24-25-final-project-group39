import numpy as np

from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Model):
    def __init__(self) -> None:
        """
        Initializes the decision tree model
        Returns:
            None
        """
        super().__init__(type="classification")
        self.model = DecisionTreeClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the decision tree to the provided training data.
        :param observations: Numpy array where each row is a data point.
        :param ground_truth: Numpy array with the corresponding labels.
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the label for each example in the input.
        :param observations: Numpy array where
        each row is a data point to predict.
        :return: Numpy array of predicted labels for each input example.
        """
        return self.model.predict(observations)
