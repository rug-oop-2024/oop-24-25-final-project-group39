"""
This module provides a function to retrieve
various classification and regression models.
"""
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.multi_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.regression.ridge import RidgeModel

from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors)
from autoop.core.ml.model.classification.decision_tree import (
    DecisionTree)
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegressionModel)
from autoop.core.ml.model.classification.linearsvc import (
    LinearSupportVectorClassifier)
from autoop.core.ml.model.classification.random_forest import (
    RandomForest)


CLASSIFICATION_MODELS = ["KNearestNeighbors", "DecisionTree",
                         "LogisticRegressionModel",
                         "LinearSupportVectorClassifier",
                         "RandomForest"]

REGRESSION_MODELS = ["MultipleLinearRegression",
                     "Lasso", "RidgeModel"]


def get_model(model_name: str) -> Model:
    """
    Factory function to retrieve a model by its name
    Args:
        model_name (str): The name of the model to retrieve
    Returns:
        Model: A Model that corresponds to the model name provided
    Raises:
        ValueError: If the model_name is not in the accepted list of models
    """
    if model_name not in REGRESSION_MODELS and \
       model_name not in CLASSIFICATION_MODELS:
        raise ValueError(f"{model_name} is not an accepted model.")
    if model_name in REGRESSION_MODELS:
        match model_name:
            case "MultipleLinearRegression":
                return MultipleLinearRegression()
            case "Lasso":
                return Lasso()
            case "RidgeModel":
                return RidgeModel()
    elif model_name in CLASSIFICATION_MODELS:
        match model_name:
            case "KNearestNeighbors":
                return KNearestNeighbors()
            case "DecisionTree":
                return DecisionTree()
            case "LogisticRegressionModel":
                return LogisticRegressionModel()
            case "LinearSupportVectorClassifier":
                return LinearSupportVectorClassifier()
            case "RandomForest":
                return RandomForest()
