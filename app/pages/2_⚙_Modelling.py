import streamlit as st

import app.core.modelling_handling as mh
from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine \
                  learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here


def check_inputs() -> bool:
    if dataset is None or input_features == [] \
     or target_feature is None or model is None or chosen_metrics == []:
        st.write("First finish the pipeline inputs")
        return False
    else:
        return True


dataset = None
input_features = []
feature_list = []
target_feature = None
model = None
chosen_model = None

st.header("Step 1. Choose Dataset")
dataset_artifact = mh.choose_dataset(datasets)
dataset = Dataset(name=dataset_artifact.name,
                  data=dataset_artifact.data,
                  asset_path=dataset_artifact.asset_path)

st.header("Step 2. Select Features")
if dataset is not None:
    feature_list = detect_feature_types(dataset)
    input_features = mh.select_input_features(dataset, feature_list)

st.header("Step 3. Select target feature")
target_feature = mh.select_target_feature(input_features, feature_list)
input_feature_names = [f.name for f in input_features]
if target_feature.name in input_feature_names:
    st.write("Target feature can not be one of the input features.")
else:
    mh.show_features(input_features, target_feature)

st.header("Step 4. Choose a model")

if target_feature.name in input_feature_names:
    st.write("Target feature can not be one of the input features.")
else:
    chosen_model = mh.choose_model(input_features, target_feature)
    if chosen_model is not None:
        model = get_model(chosen_model)

st.header("Step 5. Select data split")
train_split = mh.choose_train_split()
split = train_split / 100

st.header("Step 6. Choose Metrics")
chosen_metrics = mh.choose_metrics(chosen_model)
metric_classes = mh.metric_to_classes(chosen_metrics)

st.header("Pipeline Summary")
pipeline = None
if check_inputs() is True:
    pipeline = Pipeline(dataset=dataset,
                        metrics=metric_classes,
                        model=model,
                        input_features=input_features,
                        target_feature=target_feature,
                        split=split)
    mh.show_summary(pipeline)

st.header("Results")
if pipeline is not None:
    mh.show_results(pipeline, chosen_metrics)
    mh.save_pipeline(automl, pipeline)
