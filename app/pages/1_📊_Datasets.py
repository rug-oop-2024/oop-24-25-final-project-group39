import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from io import BytesIO
automl = AutoMLSystem.get_instance()


datasets = automl.registry.list(type="dataset")

st.write("""
         # Datasets

         Here you can manage your datasets!
         """)

options = ["Upload Dataset"]
for dataset in datasets:
    options.append(dataset)
dataset_artifact = st.selectbox("Select Dataset", options)

if dataset_artifact == "Upload Dataset":
    new_dataset = st.file_uploader("Select a dataset file", type="csv")
    if new_dataset is not None:
        df = pd.read_csv(new_dataset)
        st.write(df.head())

        if st.button(f"Save dataset '{new_dataset.name}'"):
            saved_dataset = Dataset.from_dataframe(data=df,
                                                   name=new_dataset.name,
                                                   asset_path=f"dataset/"
                                                   f"{new_dataset.name}")
            automl.registry.register(saved_dataset)
            st.write(f"Succesfully saved dataset '{new_dataset.name}'")
            st.rerun()
else:
    dataset_class = Dataset(name=dataset_artifact.name,
                            data=dataset_artifact.data,
                            asset_path=dataset_artifact.asset_path)
    st.write(dataset_class.read().head())
    dataset_to_csv = BytesIO(dataset_artifact.data)
    if st.button(f"Delete dataset '{dataset_class}'"):
        automl.registry.delete(dataset_class.id)
        st.write(f"Succesfully removed dataset '{dataset_class}'")
        st.write(f"Datasets remaining: {datasets}")


st.subheader("Currently saved datasets:")
if len(datasets) == 0:
    st.write("There are currently no saved datasets.")
for dataset in datasets:
    st.write(dataset)