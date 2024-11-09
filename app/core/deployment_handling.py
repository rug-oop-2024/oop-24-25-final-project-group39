import streamlit as st
from autoop.core.ml.dataset import Dataset


def delete_pipeline(automl, pipeline) -> None:
    automl.registry.delete(pipeline.id)
    st.rerun()
    st.write(f"Pipeline '{pipeline.name}' deleted successfully.")


def compute_results(pipeline, dataset, dataframe) -> None:
    pipeline._dataset = Dataset.from_dataframe(data=dataframe,
                                               name=dataset.name,
                                               asset_path=dataset.name)
    pipeline.execute()
    for i in range(len(pipeline._metrics)):
        st.write(f"{pipeline.metrics[i].__class__.__name__}: "
                 f"{pipeline._metrics_results[i][1]:.2f}")
