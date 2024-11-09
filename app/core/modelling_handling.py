from io import BytesIO
import streamlit as st
import pandas as pd
from typing import List
import time

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import (REGRESSION_MODELS,
                                  CLASSIFICATION_MODELS)
from autoop.core.ml.metric import METRICS, get_metric, Metric
from autoop.core.ml.pipeline import Pipeline
from app.core.system import AutoMLSystem

automl = AutoMLSystem.get_instance()


def choose_dataset(datasets: List[Dataset]) -> "Dataset":
    """
    Allows the user to select a dataset from available datasets
    and displays a preview
    Args:
        datasets (List[Dataset]): A list of available datasets
    Returns:
        Dataset: The selected dataset object
    """
    if len(datasets) == 0:
        st.write("There are no saved datasets. \
                 Please go to the Datasets page to save datasets.")
    else:
        chosen_dataset = st.selectbox("Select dataset:", datasets)
        dataset_to_csv = BytesIO(chosen_dataset.data)
        df = pd.read_csv(dataset_to_csv)
        st.write(df.head())
        return chosen_dataset


def select_input_features(data: Dataset, feature_list: List[str]) -> List[str]:
    """
    Allows the user to select input features from the available feature list
    Args:
        data (Dataset): The dataset object containing the available features
        feature_list (List[str]): A list of feature names in the dataset
    Returns:
        List[str]: A list of selected input features
    """
    if data is None:
        st.write("Please select a dataset first.")
    else:
        return st.multiselect("Select input features", feature_list)


def select_target_feature(input_features: List[str],
                          feature_list: List[str]) -> str:
    """
    Allows the user to select the target feature from
    the available input features
    Args:
        input_features (List[str]): A list of selected input features
        feature_list (List[str]): A list of feature names in the dataset
    Returns:
        str: The selected target feature
    """
    if input_features is None:
        st.write("First select input features")
    else:
        return st.selectbox("Select target feature", feature_list)


def show_features(input_features: List[str], target_feature: str) -> None:
    """
    Displays the selected input features and the target feature
    Args:
        input_features (List[str]): A list of selected input features
        target_feature (str): The selected target feature
    Returns:
        None
    """
    input_feature_names = [feature.name for feature in input_features]
    st.write("Input features: " +
             ', '.join([feature.name for feature in input_features])
             if input_feature_names != [] else "No input features selected")
    st.write("Target feature: " + target_feature.name if target_feature is not
             None else "No target feature selected")


def choose_model(input_features: List[str], target_feature: str) -> str:
    """
    Allows the user to choose a regression or classification model
    based on the selected features
    Args:
        input_features (List[str]): A list of selected input features
        target_feature (str): The selected target feature
    Returns:
        str: The name of the chosen model
    """
    if input_features is None or input_features == []:
        st.write("First select input features")
    elif target_feature is None:
        st.write("First select a target feature")
    else:
        if target_feature in input_features:
            st.write("Target feature can't be one of the input features.")
        if target_feature.type == "categorical":
            chosen_model = st.selectbox("Choose a classification model",
                                        CLASSIFICATION_MODELS)
        else:
            chosen_model = st.selectbox("Choose a regression model",
                                        REGRESSION_MODELS)
        return chosen_model


def choose_train_split() -> float:
    """
    Allows the user to select the training data split percentage
    Returns:
        float: The percentage of data to be used for training
    """
    train_split = st.slider("Training data percentage", 50, 99, 75)
    st.write("##### Updated Data Splits:")
    st.write(f"Train Split: {train_split}%")
    st.write(f"Test Split: {100 - train_split}%")
    return train_split


def choose_metrics(chosen_model: str) -> List[str]:
    """
    Allows the user to select metrics for evaluating the model
    Args:
        chosen_model (str): The model for which metrics are to be chosen
    Returns:
        List[str]: A list of selected metrics
    """
    metrics = []
    if chosen_model is None:
        st.write("First choose a model")
    elif chosen_model in CLASSIFICATION_MODELS:
        metrics = st.multiselect("Select Classification Metrics", METRICS[4:])
    elif chosen_model in REGRESSION_MODELS:
        metrics = st.multiselect("Select Regression Metrics", METRICS[:4])
    return metrics


def metric_to_classes(chosen_metrics: List[str]) -> List["Metric"]:
    """
    Converts the selected metric names into Metric class objects
    Args:
        chosen_metrics (List[str]): A list of selected metric names
    Returns:
        List[Metric]: A list of corresponding Metric objects
    """
    metric_models = []
    if chosen_metrics != []:
        for i in range(len(chosen_metrics)):
            metric_models.append(get_metric(chosen_metrics[i]))
            st.write(chosen_metrics[i])
    return metric_models


def show_summary(pipeline: Pipeline) -> None:
    """
    Displays a summary of the selected dataset, features, model,
    data split, and metrics
    Args:
         pipeline (Pipeline): The pipeline object containing
         the selected configurations
    Returns:
        None
    """
    col1, col2 = st.columns([2, 7])
    col1.write("Dataset")
    col1.write("Input Features")
    col1.write("Target Feature")
    col1.write("Model")
    col1.write("Data Split")
    col1.write("Metrics")
    col2.write(pipeline._dataset.name)
    col2.write(', '.join([pipeline.name for
                          pipeline in pipeline._input_features]))
    col2.write(pipeline._target_feature.name)
    col2.write(pipeline.model.__class__.__name__)
    col2.write(f"Train: {pipeline._split*100}% // Test: "
               f"{(1 - pipeline._split) * 100}%")
    col2.write(', '.join([metric.__class__.__name__
                          for metric in pipeline._metrics]))


def show_results(pipeline: Pipeline, chosen_metrics: List[str]) -> None:
    """
    Displays the results of the selected metrics
    Args:
        pipeline (Pipeline): The pipeline object to execute
        chosen_metrics (List[str]): A list of selected metrics for evaluation
    Returns:
        None
    """
    if st.button("Compute results"):
        pipeline.execute()
        st.write("#### Metrics:")
        calculating = st.caption("Calculating...")
        progress_bar = st.progress(0)
        for percentage in range(101):
            progress_bar.progress(percentage)
            time.sleep(0.03)
        time.sleep(0.5)
        progress_bar.empty()
        calculating.empty()
        for i in range(len(chosen_metrics)):
            st.write(f"{chosen_metrics[i]}: \
                     {pipeline._metrics_results[i][1]:.2f}")


def save_pipeline(automl: AutoMLSystem, pipeline: Pipeline) -> None:
    """
    Saves the pipeline with a name and version
    Args:
        automl (AutoMLSystem): The AutoML system instance
        pipeline (Pipeline): The pipeline object to be saved
    Returns:
        None
    """
    st.header("Save Pipeline")
    name_pipeline = st.text_input("Enter a name for this pipeline")
    version = st.text_input("Enter a version for this pipeline")
    if st.button("Save Pipeline"):
        artifact_pipeline = pipeline.to_artifact(name_pipeline, version)
        automl.registry.register(artifact_pipeline)
        st.write("Sucessfully saved pipeline")
