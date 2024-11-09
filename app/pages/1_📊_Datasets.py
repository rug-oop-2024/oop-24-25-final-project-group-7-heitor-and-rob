import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")

# Display the list of datasets
st.subheader("Available Datasets")
for dataset in datasets:
    st.write(f"Name: {dataset.name}, Type: {dataset.type}")

# Add a new dataset
st.subheader("Add New Dataset")
new_dataset_name = st.text_input("Dataset Name")
new_dataset_type = st.text_input("Dataset Type")
if st.button("Add Dataset"):
    new_dataset = Dataset(name=new_dataset_name, type=new_dataset_type)
    automl.registry.add(new_dataset)
    st.success(f"Dataset {new_dataset_name} added successfully!")

# Delete a dataset
st.subheader("Delete Dataset")
dataset_to_delete = st.selectbox("Select Dataset to Delete", [d.name for d in datasets])
if st.button("Delete Dataset"):
    dataset = next(d for d in datasets if d.name == dataset_to_delete)
    automl.registry.remove(dataset)
    st.success(f"Dataset {dataset_to_delete} deleted successfully!")

# Update a dataset
st.subheader("Update Dataset")
dataset_to_update = st.selectbox("Select Dataset to Update", [d.name for d in datasets])
updated_dataset_name = st.text_input("Updated Dataset Name")
updated_dataset_type = st.text_input("Updated Dataset Type")
if st.button("Update Dataset"):
    dataset = next(d for d in datasets if d.name == dataset_to_update)
    dataset.name = updated_dataset_name
    dataset.type = updated_dataset_type
    automl.registry.update(dataset)
    st.success(f"Dataset {dataset_to_update} updated successfully!")
