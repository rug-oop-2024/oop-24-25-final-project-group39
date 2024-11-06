import streamlit as st
import pandas as pd
import time
from io import BytesIO

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import METRICS, get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification import DecisionTree, KNearestNeighbors, LogisticRegressionModel
from autoop.core.ml.model.regression import MultipleLinearRegression, RidgeModel
from autoop.core.ml.pipeline import Pipeline


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here


dataset = None
input_features = None
input_feature_names = []
target_feature = None
model_list = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression",
              "Multiple Linear Regression", "Ridge"]
models = [DecisionTree(), KNearestNeighbors(), LogisticRegressionModel(), MultipleLinearRegression(), RidgeModel()]
model = None

st.header("Step 1. Choose Dataset")
if len(datasets) == 0:
    st.write("There are no saved datasets. Please go to the Datasets page to save datasets.")
else:
    dataset = st.selectbox("Select dataset:", datasets)
    dataset_to_csv = BytesIO(dataset.data)
    df = pd.read_csv(dataset_to_csv)
    st.write(df.head())

st.header("Step 2. Select Features")

if dataset is None:
    st.write("Please select a dataset first.")
else:
    feature_list = detect_feature_types(dataset)
    input_features = st.multiselect("Select input features", feature_list)
    input_feature_names = [feature.name for feature in input_features]
    target_feature = st.selectbox("Select target feature", feature_list)
    if input_features is not None and target_feature is not None:
        if target_feature.name in input_feature_names:
            st.write("Target feature can not be one of the input features.")
        else:
            st.write(f"Input features: {', '.join([feature.name for feature in input_features]) \
                                        if input_feature_names != [] else "No input features selected"}")
            st.write(f"Target feature: {target_feature.name}")

            st.header("Step 3. Choose a model")
            if input_features is None or input_features == []:
                st.write("First select input features")
            elif target_feature is None:
                st.write("First select a target feature")
            else:
                if target_feature.type == "categorical":
                    chosen_model = st.selectbox("Choose a classification model", model_list[:3])
                else:
                    chosen_model = st.selectbox("Choose a regression model", model_list[3:])
                for i in range(len(model_list)):
                    if chosen_model == model_list[i]:
                        model = models[i]
                st.write(chosen_model)
                st.write(model)


st.header("Step 4. Select data split")
train_split = st.slider("Training data percentage", 50, 99, 75)
st.write("##### Updated Data Splits:")
st.write(f"Train Split: {train_split}%")
st.write(f"Test Split: {100 - train_split}%")
split = train_split / 100




st.header("Step 5. Choose Metrics")
chosen_metrics = []
metric_models = []
if model is None:
    st.write("First choose a model")
elif model in models[:3]:
    chosen_metrics = st.multiselect("Select Classification Metrics", METRICS[4:])
elif model in models[3:]:
    chosen_metrics = st.multiselect("Select Regression Metrics", METRICS[:4])
if chosen_metrics != []:
    metric_models = []
    for i in range(len(chosen_metrics)):
        metric_models.append(get_metric(chosen_metrics[i]))
        st.write(chosen_metrics[i])
            
st.header("Pipeline Summary")

col1, col2 = st.columns([2, 7])
col1.write("Dataset")
col1.write("Input Features")
col1.write("Target Feature")
col1.write("Model")
col1.write("Data Split")
col1.write("Metrics")
col2.write(dataset.name if dataset is not None else "No dataset selected")
col2.write(', '.join(input_feature_names) if input_feature_names != [] else "No input features selected")
col2.write(target_feature.name if target_feature is not None else "No target feature selected")
col2.write(chosen_model if model is not None else "No model selected")
col2.write(f"Train: {train_split}% // Test: {100 - train_split}%")
col2.write(', '.join(chosen_metrics) if chosen_metrics != [] else "No metrics selected")


st.header("Results")
if dataset is None or input_features is None or input_feature_names == [] \
or target_feature is None or model is None or chosen_metrics == []:
    st.write("First finish the pipeline inputs")
else:
    if st.button("Compute results"):
        results = Pipeline(dataset=dataset,
                        metrics=metric_models,
                        model=model,
                        input_features=input_features,
                        target_feature=target_feature,
                        split=split)
        results.execute()
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
            st.write(f"{chosen_metrics[i]}: {results._metrics_results[i][1]:.2f}")
    st.header("Save Pipeline")
    name_pipeline = st.text_input("Enter a name for this pipeline")
    if st.button("Save Pipeline"):
        pass


