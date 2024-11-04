import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from io import BytesIO
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here


dataset = None
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
            st.write(f"Input features: {', '.join([feature.name for feature in input_features])}")
            st.write(f"Target feature: '{target_feature.name}'")

