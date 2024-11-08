import streamlit as st
import pandas as pd
import time
from io import BytesIO
from typing import List

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import METRICS, get_metric, Metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification import (
    DecisionTree, KNearestNeighbors, LogisticRegressionModel)
from autoop.core.ml.model.regression import (
    MultipleLinearRegression, RidgeModel)
from autoop.core.ml.model.model import Model
from autoop.core.ml.pipeline import Pipeline


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine \
                  learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here


def choose_dataset() -> "Dataset":
    if len(datasets) == 0:
        st.write("There are no saved datasets. \
                 Please go to the Datasets page to save datasets.")
    else:
        chosen_dataset = st.selectbox("Select dataset:", datasets)
        dataset_to_csv = BytesIO(chosen_dataset.data)
        df = pd.read_csv(dataset_to_csv)
        st.write(df.head())
        return chosen_dataset


def select_input_features(data) -> List[str]:
    if data is None:
        st.write("Please select a dataset first.")
    else:
        return st.multiselect("Select input features", feature_list)


def select_target_feature(feature_list):
    if input_features is None:
        st.write("First select input features")
    else:
        return st.selectbox("Select target feature", feature_list)


def show_features():
    st.write(f"Input features: {', '.join([feature.name for
                                           feature in input_features])
                                if input_feature_names != []
                                else "No input features selected"}")
    st.write(f"Target feature: {target_feature.name if target_feature is
                                not None else "No target feature selected"}")


def choose_model() -> "Model":
    if input_features is None or input_features == []:
        st.write("First select input features")
    elif target_feature is None:
        st.write("First select a target feature")
    else:
        if target_feature in input_features:
            st.write("Target feature can't be one of the input features.")
        if target_feature.type == "categorical":
            chosen_model = st.selectbox("Choose a classification model",
                                        model_list[:3])
        else:
            chosen_model = st.selectbox("Choose a regression model",
                                        model_list[3:])
        return chosen_model


def model_to_class(model: str) -> "Model":
    for i in range(len(model_list)):
        if model == model_list[i]:
            return models[i]


def choose_train_split() -> float:
    train_split = st.slider("Training data percentage", 50, 99, 75)
    st.write("##### Updated Data Splits:")
    st.write(f"Train Split: {train_split}%")
    st.write(f"Test Split: {100 - train_split}%")
    return train_split


def choose_metrics() -> List[str]:
    metrics = []
    if model is None:
        st.write("First choose a model")
    elif model in models[:3]:
        metrics = st.multiselect("Select Classification Metrics", METRICS[4:])
    elif model in models[3:]:
        metrics = st.multiselect("Select Regression Metrics", METRICS[:4])
    return metrics


def metric_to_classes(chosen_metrics: List[str]) -> List["Metric"]:
    metric_models = []
    if chosen_metrics != []:
        for i in range(len(chosen_metrics)):
            metric_models.append(get_metric(chosen_metrics[i]))
            st.write(chosen_metrics[i])
    return metric_models


def show_summary():
    col1, col2 = st.columns([2, 7])
    col1.write("Dataset")
    col1.write("Input Features")
    col1.write("Target Feature")
    col1.write("Model")
    col1.write("Data Split")
    col1.write("Metrics")
    col2.write(dataset.name if dataset is not None else "No dataset selected")
    col2.write(', '.join(input_feature_names)
               if input_feature_names != [] else "No input features selected")
    col2.write(target_feature.name
               if target_feature is not None else "No target feature selected")
    col2.write(chosen_model if model is not None else "No model selected")
    col2.write(f"Train: {train_split}% // Test: {100 - train_split}%")
    col2.write(', '.join(chosen_metrics)
               if chosen_metrics != [] else "No metrics selected")


def check_inputs() -> bool:
    if dataset is None or input_features == [] or input_feature_names == [] \
     or target_feature is None or model is None or chosen_metrics == []:
        st.write("First finish the pipeline inputs")
        return False
    else:
        return True


def show_results():
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


def save_pipeline():
    st.header("Save Pipeline")
    name_pipeline = st.text_input("Enter a name for this pipeline")
    version = st.text_input("Enter a version for this pipeline")
    if st.button("Save Pipeline"):
        artifact_pipeline = pipeline.to_artifact(name_pipeline, version)
        automl.registry.register(artifact_pipeline)
        st.write("Sucessfully saved pipeline")


dataset = None
input_features = []
feature_list = []
target_feature = None
model_list = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression",
              "Multiple Linear Regression", "Ridge"]
models = [DecisionTree(), KNearestNeighbors(), LogisticRegressionModel(),
          MultipleLinearRegression(), RidgeModel()]
model = None

st.header("Step 1. Choose Dataset")
dataset = choose_dataset()

st.header("Step 2. Select Features")
if dataset is not None:
    feature_list = detect_feature_types(dataset)
    input_features = select_input_features(dataset)
input_feature_names = [feature.name for feature in input_features]
new_feature_list = [feature for feature
                    in feature_list if feature not in input_features]

st.header("Step 3. Select target feature")
target_feature = select_target_feature(new_feature_list)
show_features()

st.header("Step 4. Choose a model")
chosen_model = choose_model()
model = model_to_class(chosen_model)

st.header("Step 5. Select data split")
train_split = choose_train_split()
split = train_split / 100

st.header("Step 6. Choose Metrics")
chosen_metrics = choose_metrics()
metric_classes = metric_to_classes(chosen_metrics)

st.header("Pipeline Summary")
show_summary()

st.header("Results")
if check_inputs() is True:
    pipeline = Pipeline(dataset=dataset,
                        metrics=metric_classes,
                        model=model,
                        input_features=input_features,
                        target_feature=target_feature,
                        split=split)
    show_results()
    save_pipeline()
