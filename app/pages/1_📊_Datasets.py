import streamlit as st
import pandas as pd
from copy import deepcopy

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from io import BytesIO
automl = AutoMLSystem.get_instance()


datasets = automl.registry.list(type="dataset")

# your code here

def give_options():
    options = ["Upload Dataset"]
    for dataset in datasets:
        options.append(dataset)
    return st.selectbox("Select Dataset", options)

def show_dataset(dataset) -> pd.DataFrame:
    dataframe = pd.read_csv(dataset)
    st.write(dataframe.head())
    return dataframe

def save_dataset(dataset, dataframe):
    saved_dataset = Dataset.from_dataframe(data=dataframe, name=dataset.name, asset_path=dataset.name)
    automl.registry.register(saved_dataset)
    st.write(f"Successfully saved dataset '{dataset.name}'")
    st.rerun()

def delete_dataset(dataset):
    automl.registry.delete(dataset.id)
    st.write(f"Successfully removed dataset '{dataset}'")
    st.rerun()

def show_saved_datasets():
    st.subheader("Currently saved datasets:")
    if len(datasets) == 0:
        st.write("There are currently no saved datasets.")
    for dataset in datasets:
        st.write(dataset)

st.header("Datasets") 
st.write("#### Here you can manage your datasets!")

chosen_dataset = give_options()

if chosen_dataset == "Upload Dataset":
    new_dataset = st.file_uploader("Select a dataset file", type = "csv")
    if new_dataset is not None:
        df = show_dataset(new_dataset)
        if st.button(f"Save dataset '{new_dataset.name}'"):
            save_dataset(new_dataset, df)
else:
    dataset_to_csv = BytesIO(chosen_dataset.data)
    show_dataset(dataset_to_csv)
    if st.button(f"Delete dataset '{chosen_dataset}'"):
        delete_dataset(chosen_dataset)

show_saved_datasets()

