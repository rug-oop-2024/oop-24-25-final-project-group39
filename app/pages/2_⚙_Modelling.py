import streamlit as st

import app.core.modelling_handling as mh
from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Writes helper text in a Streamlit app with custom styling
    Args:
        text (str): The helper text to be displayed
    Returns:
        None
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine \
                  learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here


def check_inputs() -> bool:
    """
    Checks if all necessary inputs for the pipeline are provided
    Returns:
        bool: True if all inputs are provided, otherwise False.
    """
    if dataset is None or input_features == [] or target_feature is None \
            or model is None or chosen_metrics == []:
        st.write("First finish the pipeline inputs")
        return False
    else:
        return True


dataset = None
input_features = []
feature_list = []
target_feature = None
model = None

st.header("Step 1. Choose Dataset")
dataset = mh.choose_dataset(datasets)

st.header("Step 2. Select Features")
if dataset is not None:
    feature_list = detect_feature_types(dataset)
    input_features = mh.select_input_features(dataset, feature_list)
new_feature_list = [feature for feature
                    in feature_list if feature not in input_features]

st.header("Step 3. Select target feature")
target_feature = mh.select_target_feature(input_features, new_feature_list)
mh.show_features(input_features, target_feature)

st.header("Step 4. Choose a model")

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
