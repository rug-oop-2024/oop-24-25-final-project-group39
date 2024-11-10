import numpy as np

from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Model):
    """Decision tree model for classification using scikit-learn"""
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
        Fits the model to the training data
        Args:
            observations (np.ndarray): Training data where
            each row is a data point
            ground_truth (np.ndarray): Target labels for each data
            point in the training set
        Returns:
            None
        """
        self.model.fit(observations, ground_truth)
        self.parameters = {
            "feature_importances": self.model.feature_importances_.tolist(),
            "max_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves()
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the label for each example in the input.
        :param observations: Numpy array where
        each row is a data point to predict.
        :return: Numpy array of predicted labels for each input example.
        """
        return self.model.predict(observations)
