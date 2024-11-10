import streamlit as st
import pickle
import pandas as pd
from app.core.system import AutoMLSystem
import app.core.deployment_handling as dh
import app.core.modelling_handling as mh


automl = AutoMLSystem.get_instance()

pipeline_list = automl.registry.list(type="pipeline")

st.header("Pipelines")
if pipeline_list == []:
    st.write("No saved pipelines found.")
else:
    chosen_pipeline = None
    selected_pipeline = st.selectbox("Choose a pipeline",
                                     [f"{pipeline.name} // {pipeline.version}"
                                      for pipeline in pipeline_list])

    for i in range(len(pipeline_list)):
        if selected_pipeline == \
           f"{pipeline_list[i].name} // {pipeline_list[i].version}":
            chosen_pipeline = pipeline_list[i]

    st.write(chosen_pipeline._name)
    pipeline = pickle.loads(chosen_pipeline._data)

    st.header("Summary")
    mh.show_summary(pipeline)

    if st.button("Delete Pipeline"):
        dh.delete_pipeline(automl, chosen_pipeline)

    dataset = st.file_uploader("Select a dataset file", type="csv")
    if dataset is not None:
        df = pd.read_csv(dataset)
        st.write(df.head())
        if st.button("Compute Results"):
            dh.compute_results(pipeline, dataset, df)