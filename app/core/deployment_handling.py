import streamlit as st
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from app.core.system import AutoMLSystem

automl = AutoMLSystem.get_instance()


def delete_pipeline(automl: AutoMLSystem, pipeline: Pipeline) -> None:
    """
    Deletes the pipeline from the AutoML system
    Args:
        automl (AutoMLSystem): The AutoML system instance that
        manages the pipeline
        pipeline (Pipeline): The pipeline object to be deleted
    Returns:
        None
    """
    automl.registry.delete(pipeline.id)
    st.rerun()
    st.write(f"Pipeline '{pipeline.name}' deleted successfully.")


def compute_results(pipeline: Pipeline, dataset: Dataset,
                    dataframe: pd.DataFrame) -> None:
    """
    Computes the results for the given pipeline using a dataset and dataframe,
    and displays the selected metrics
    Args:
        pipeline (Pipeline): The pipeline to execute on the data
        dataset (Dataset): The dataset object containing metadata
        dataframe (pd.DataFrame): The data in the form of a DataFrame
    Returns:
        None
    """
    pipeline._dataset = Dataset.from_dataframe(data=dataframe,
                                               name=dataset.name,
                                               asset_path=dataset.name)
    pipeline.execute()
    for i in range(len(pipeline._metrics)):
        st.write(f"{pipeline.metrics[i].__class__.__name__}: "
                 f"{pipeline._metrics_results[i][1]:.2f}")
