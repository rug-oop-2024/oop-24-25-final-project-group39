import numpy as np

from autoop.core.ml.model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):
    """Random Forest model for classification using scikit-learn"""

    def __init__(self, n_estimators: int = 100,
                 max_depth: int = None, random_state: int = None) -> None:
        """
        Initializes the Random Forest classifier
        Args:
            n_estimators (int): The number of trees in the forest,
            defaults to 100
            max_depth (int): The maximum depth of the trees,
            defaults to None
            random_state (int): Controls the randomness of the estimator,
            defaults to None
        Returns:
            None
        """
        super().__init__(type="classification",
                         parameters={"n_estimators": n_estimators,
                                     "max_depth": max_depth,
                                     "random_state": random_state})
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=random_state)

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

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts labels for each observation in the input
        Args:
            observations (np.ndarray): Input data features for prediction
        Returns:
            np.ndarray: Predicted labels for each input observation
        """
        return self.model.predict(observations)
