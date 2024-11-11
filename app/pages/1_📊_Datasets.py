import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")


def get_datasets() -> list:
    """
    Retrieve the list of datasets from the registry.

    :return: List of datasets
    :rtype: list
    """
    return automl.registry.list(type="dataset")


datasets = get_datasets()

st.subheader("Available Datasets")
if datasets:
    for dataset in datasets:
        st.write(f"Name: {dataset.name}, Type: {dataset.type}")
else:
    st.write("No datasets available.")

st.subheader("Upload new Dataset")
uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file).dropna()
    st.write("### Preview of the uploaded dataset:")
    st.write(data.head())

    dataset_name = st.text_input("Enter a name for the dataset")

    if dataset_name:
        asset_path = f"dataset/{dataset_name}.csv"

        new_dataset = Dataset.from_dataframe(
            data=data,
            name=dataset_name,
            asset_path=asset_path,
            version="1.0.0",
        )

        if st.button("Save Dataset"):
            automl.registry.register(new_dataset)
            st.success(f"Dataset '{dataset_name}' uploaded and saved "
                       "successfully!")
            st.rerun()
    else:
        st.warning("Please enter a name for the dataset.")
