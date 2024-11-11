import streamlit as st
import pandas as pd
from typing import List, Dict
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Modelling", page_icon="🚀")

saved_pipelines = automl.registry.list(type="pipeline")


def get_saved_pipelines() -> List[Dict[str, ]]:
    return saved_pipelines


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

        st.subheader("Pipeline Summary")
        st.write(f"**Name**: {selected_pipeline['name']}")
        st.write(f"**Version**: {selected_pipeline['version']}")
        st.write(f"**Type**: {selected_pipeline['type']}")
        st.write(f"**ID**: {selected_pipeline['id']}")
        st.write(f"**Tags**: {', '.join(selected_pipeline['tags'])}")

        st.subheader("Metadata")
        st.write(
            f"- **Input Features**: {', '.join(selected_pipeline['metadata']['input_features'])}")
        st.write(
            f"- **Target Feature**: {selected_pipeline['metadata']['target_feature']}")
        st.write(f"- **Model**: {selected_pipeline['metadata']['model']}")
        st.write(
            f"- **Task Type**: {selected_pipeline['metadata']['task_type']}")
        st.write(
            f"- **Split Ratio**: {selected_pipeline['metadata']['split_ratio']}")
        st.write(
            f"- **Metrics**: {', '.join(selected_pipeline['metadata']['metrics'])}")

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
                predictions = data[selected_pipeline['metadata']['input_features']].apply(
                    lambda x: 'Prediction', axis=1)
                st.write("Predicted Results:")
                st.dataframe(predictions)
else:
    st.write("No saved pipelines found.")
