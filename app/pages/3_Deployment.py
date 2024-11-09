import streamlit as st
import pickle
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

pipeline_list = automl.registry.list(type="pipeline")


def show_pipeline_summary() -> None:
    st.header("Pipeline Summary")
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
    col2.write(pipeline.model._class.name_)
    col2.write(f"Train: {pipeline._split*100}% // Test: "
               f"{(1 - pipeline._split) * 100}%")
    col2.write(', '.join([metric.__class__.__name__
                          for metric in pipeline._metrics]))


def delete_pipeline() -> None:
    automl.registry.delete(chosen_pipeline.id)
    st.rerun()
    st.write(f"Pipeline '{selected_pipeline}' deleted successfully.")


def compute_results() -> None:
    pipeline._dataset = Dataset.from_dataframe(data=df,
                                               name=dataset.name,
                                               asset_path=dataset.name)
    pipeline.execute()
    for i in range(len(pipeline._metrics)):
        st.write(f"{pipeline.metrics[i].__class__.__name__}: "
                 f"{pipeline._metrics_results[i][1]:.2f}")


st.header("Pipelines")
if pipeline_list == []:
    st.write("No saved pipelines found.")
else:
    selected_pipeline = st.selectbox("Choose a pipeline",
                                     [f"{pipeline.name} // {pipeline.version}"
                                      for pipeline in pipeline_list])

    for i in range(len(pipeline_list)):
        if selected_pipeline == f"{pipeline_list[i].name} // \
        {pipeline_list[i].version}":
            chosen_pipeline = pipeline_list[i]

    st.write(chosen_pipeline._name)
    pipeline = pickle.loads(chosen_pipeline._data)

    show_pipeline_summary()

    if st.button("Delete Pipeline"):
        delete_pipeline()

    dataset = st.file_uploader("Select a dataset file", type="csv")
    if dataset is not None:
        df = pd.read_csv(dataset)
        st.write(df.head())
        if st.button("Compute Results"):
            compute_results()
