from io import BytesIO
import streamlit as st
import pandas as pd
from typing import List
import time

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import (REGRESSION_MODELS,
                                  CLASSIFICATION_MODELS)
from autoop.core.ml.metric import METRICS, get_metric, Metric


def choose_dataset(datasets) -> "Dataset":
    if len(datasets) == 0:
        st.write("There are no saved datasets. \
                 Please go to the Datasets page to save datasets.")
    else:
        chosen_dataset = st.selectbox("Select dataset:", datasets)
        dataset_to_csv = BytesIO(chosen_dataset.data)
        df = pd.read_csv(dataset_to_csv)
        st.write(df.head())
        return chosen_dataset


def select_input_features(data, feature_list) -> List[str]:
    if data is None:
        st.write("Please select a dataset first.")
    else:
        return st.multiselect("Select input features", feature_list)


def select_target_feature(input_features, feature_list) -> str:
    if input_features is None:
        st.write("First select input features")
    else:
        return st.selectbox("Select target feature", feature_list)


def show_features(input_features, target_feature) -> None:
    input_feature_names = [feature.name for feature in input_features]
    st.write("Input features: " +
             ', '.join([feature.name for feature in input_features])
             if input_feature_names != [] else "No input features selected")
    st.write("Target feature: " + target_feature.name if target_feature is not
             None else "No target feature selected")


def choose_model(input_features, target_feature) -> str:
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
    train_split = st.slider("Training data percentage", 50, 99, 75)
    st.write("##### Updated Data Splits:")
    st.write(f"Train Split: {train_split}%")
    st.write(f"Test Split: {100 - train_split}%")
    return train_split


def choose_metrics(chosen_model) -> List[str]:
    metrics = []
    if chosen_model is None:
        st.write("First choose a model")
    elif chosen_model in CLASSIFICATION_MODELS:
        metrics = st.multiselect("Select Classification Metrics", METRICS[5:])
    elif chosen_model in REGRESSION_MODELS:
        metrics = st.multiselect("Select Regression Metrics", METRICS[:5])
    return metrics


def metric_to_classes(chosen_metrics: List[str]) -> List["Metric"]:
    metric_models = []
    if chosen_metrics != []:
        for i in range(len(chosen_metrics)):
            metric_models.append(get_metric(chosen_metrics[i]))
            st.write(chosen_metrics[i])
    return metric_models


def show_summary(pipeline) -> None:
    col1, col2 = st.columns([2, 7])
    col1.write("Dataset")
    col1.write("Input Features")
    col1.write("Target Feature")
    col1.write("Model")
    col1.write("Data Split")
    col1.write("Metrics")
    col2.write(pipeline._dataset.name)
    col2.write(', '.join([feature.name for
                          feature in pipeline._input_features]))
    col2.write(pipeline._target_feature.name)
    col2.write(pipeline.model.__class__.__name__)
    col2.write(f"Train: {pipeline._split*100}% // Test: "
               f"{(1 - pipeline._split) * 100}%")
    col2.write(', '.join([metric.__class__.__name__
                          for metric in pipeline._metrics]))


def show_results(pipeline, chosen_metrics) -> None:
    if st.button("Compute results"):
        results = pipeline.execute()
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
        with st.expander("", expanded=True):
            st.write("### Predictions:")
            st.write(results['predictions'])


def save_pipeline(automl, pipeline) -> None:
    st.header("Save Pipeline")
    name_pipeline = st.text_input("Enter a name for this pipeline")
    version = st.text_input("Enter a version for this pipeline")
    if st.button("Save Pipeline"):
        artifact_pipeline = pipeline.to_artifact(name_pipeline, version)
        automl.registry.register(artifact_pipeline)
        st.write("Sucessfully saved pipeline")
