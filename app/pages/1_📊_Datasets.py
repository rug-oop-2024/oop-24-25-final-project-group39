import streamlit as st
import pandas as pd
# from copy import deepcopy

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from io import BytesIO
automl = AutoMLSystem.get_instance()


datasets = automl.registry.list(type="dataset")

# your code here


st.write("""
         # Datasets

         Here you can manage your datasets!
         """)

options = ["Upload Dataset"]
for dataset in datasets:
    options.append(dataset)
dataset = st.selectbox("Select Dataset", options)
if dataset == "Upload Dataset":
    new_dataset = st.file_uploader("Select a dataset file", type="csv")
    if new_dataset is not None:
        df = pd.read_csv(new_dataset)
        st.write(df.head())
        if st.button(f"Save dataset '{new_dataset.name}'"):
            saved_dataset = Dataset.from_dataframe(data=df,
                                                   name=new_dataset.name,
                                                   asset_path=new_dataset.name)
            automl.registry.register(saved_dataset)
            st.write(f"Succesfully saved dataset '{new_dataset.name}'")
else:
    st.dataframe(dataset.read())
    dataset_to_csv = BytesIO(dataset.data)
    df = pd.read_csv(dataset_to_csv)
    st.write(df.head())
    if st.button(f"Delete dataset '{dataset}'"):
        automl.registry.delete(dataset.id)
        st.write(f"Succesfully removed dataset '{dataset}'")
        st.write(f"Datasets remaining: {datasets}")

st.subheader("Currently saved datasets:")
if len(datasets) == 0:
    st.write("There are currently no saved datasets.")
for dataset in datasets:
    st.write(dataset)
