import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")


def get_datasets():
    return automl.registry.list(type="dataset")


datasets = get_datasets()

# Display the list of datasets
st.subheader("Available Datasets")
if datasets:
    for dataset in datasets:
        st.write(f"Name: {dataset.name}, Type: {dataset.type}")
else:
    st.write("No datasets available.")

# Add a new dataset
st.subheader("Add New Dataset")
new_dataset_name = st.text_input("Dataset Name")
new_dataset_type = st.text_input("Dataset Type")
if st.button("Add Dataset"):
    if new_dataset_name and new_dataset_type:
        if any(d.name == new_dataset_name for d in datasets):
            st.error(f"Dataset {new_dataset_name} already exists!")
        else:
            new_dataset = Dataset(name=new_dataset_name, type=new_dataset_type)
            automl.registry.add(new_dataset)
            st.success(f"Dataset {new_dataset_name} added successfully!")
            st.rerun()
    else:
        st.error("Please enter both dataset name and type.")

# Delete a dataset
st.subheader("Delete Dataset")
if datasets:
    dataset_to_delete = st.selectbox("Select Dataset to Delete", [
                                     d.name for d in datasets])
    if st.button("Delete Dataset"):
        dataset = next(
            (d for d in datasets if d.name == dataset_to_delete), None)
        if dataset:
            automl.registry.remove(dataset)
            st.success(f"Dataset {dataset_to_delete} deleted successfully!")
            st.rerun()
else:
    st.write("No datasets available to delete.")

# Update a dataset
st.subheader("Update Dataset")
if datasets:
    dataset_to_update = st.selectbox("Select Dataset to Update", [
                                     d.name for d in datasets])
    updated_dataset_name = st.text_input("Updated Dataset Name")
    updated_dataset_type = st.text_input("Updated Dataset Type")
    if st.button("Update Dataset"):
        if updated_dataset_name and updated_dataset_type:
            dataset = next(
                (d for d in datasets if d.name == dataset_to_update), None)
            if dataset:
                dataset.name = updated_dataset_name
                dataset.type = updated_dataset_type
                automl.registry.update(dataset)
                st.success(
                    f"Dataset {dataset_to_update} updated successfully!")
                st.rerun()
        else:
            st.warning(
                "Please enter both an updated dataset name and type for the dataset.")
else:
    st.write("No datasets available to update.")

# Upload a dataset
st.subheader("Upload new Dataset")
uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of the uploaded dataset:")
    st.write(data.head())

    dataset_name = st.text_input("Enter a name for the dataset")

    if dataset_name:
        asset_path = f"dataset/{dataset_name}.csv"

        new_dataset = Dataset.from_dataframe(
            data=data,
            name=dataset_name,
            asset_path=asset_path,
            version="1.0.0"
        )

        if st.button("Save Dataset"):
            automl.registry.register(new_dataset)
            st.success(
                "Dataset '{dataset_name}' uploaded and saved successfully!")
            st.rerun()
    else:
        st.warning("Please enter a name for the dataset.")
