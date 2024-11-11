import streamlit as st
import pandas as pd
import pickle
import os
from typing import List, Dict
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Deployment", page_icon="✈️")


def write_helper_text(text: str) -> None:
    """
    Write helper text in a styled format.

    :param text: The text to display as helper text.
    :type text: str
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


pipeline_dir = "./assets/objects/pipelines"
saved_pipelines = automl.storage.list(pipeline_dir)


def get_saved_pipelines() -> List[Dict[str, ]]:
    pipelines = []
    for file_name in os.listdir(pipeline_dir):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(pipeline_dir, file_name)
            try:
                with open(file_path, 'rb') as file:
                    pipeline_data = pickle.load(file)
                    pipelines.append({
                        "name": file_name,
                        "path": file_path,
                        "data": pipeline_data
                    })
            except Exception as e:
                st.error(f"Error loading pipeline '{file_name}': {str(e)}")
    return pipelines


st.title("Pipeline Management")
st.write("This page allows you to view, load, and use saved pipelines for predictions.")

st.subheader("Available Pipelines")
pipelines = get_saved_pipelines()
pipeline_names = [pipeline['name'] for pipeline in pipelines]

if pipeline_names:
    selected_pipeline_name = st.selectbox(
        "Select a pipeline to load", pipeline_names)

    if selected_pipeline_name:
        selected_pipeline = next(
            pipeline for pipeline in pipelines if pipeline['name'] == selected_pipeline_name
        )

        pipeline_data = selected_pipeline['data']

        st.subheader("Pipeline Summary")
        st.write(f"**Name**: {selected_pipeline['name']}")
        st.write(f"**Model Type**: {pipeline_data['model'].type}")
        st.write(
            f"**Input Features**: {[feature.name for feature in pipeline_data['input_features']]}")
        st.write(f"**Target Feature**: {pipeline_data['target_feature'].name}")
        st.write(f"**Split Ratio**: {pipeline_data['split']}")
        st.write(
            f"**Metrics**: {[str(metric) for metric in pipeline_data['metrics']]}")

        st.subheader("Upload CSV for Predictions")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            if st.button("Perform Prediction"):
                # Replace this with actual model prediction logic
                st.write("Performing prediction on uploaded data...")
                st.success("Predictions completed successfully!")
                # Dummy prediction logic
                predictions = data[[feature.name for feature in pipeline_data['input_features']]].apply(
                    lambda x: 'Prediction', axis=1
                )
                st.write("Predicted Results:")
                st.dataframe(predictions)
else:
    st.write("No saved pipelines found.")
