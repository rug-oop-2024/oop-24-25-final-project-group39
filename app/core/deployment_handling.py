import streamlit as st
from autoop.core.ml.dataset import Dataset


def delete_pipeline(automl, pipeline) -> None:
    """
    Deletes the specified pipeline
    Args:
        automl: The AutoML system instance that manages the pipeline
        pipeline: The pipeline object to be deleted
    Returns:
        None
    """
    automl.registry.delete(pipeline.id)
    st.rerun()
    st.write(f"Pipeline '{pipeline.name}' deleted successfully.")


def compute_results(pipeline, dataset, dataframe) -> None:
    """
    Computes the results for the given pipeline using a dataset and dataframe,
    and displays the metrics
    Args:
        pipeline: The pipeline to execute on the provided data
        dataset: The dataset object containing metadata
        dataframe: The data in the form of a DataFrame
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
