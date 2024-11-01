
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
    features = []
    data = dataset.read()
    for column in data.columns:
        if data[column].dtype == 'float64' or \
           data[column].dtype == 'int64':
            features.append(Feature(column, 'numerical'))
        else:
            features.append(Feature(column, 'categorical'))
    raise NotImplementedError("This should be implemented by you.")
