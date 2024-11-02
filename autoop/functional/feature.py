
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    current_data = dataset.read()
    feature_list = []
    for column in current_data.columns:
        if current_data[column].dtype == "float64" or \
           current_data[column].dtype == "int64":
            feature_list.append(Feature(column, 'numerical'))
        else:
            feature_list.append(Feature(column, 'categorical'))
