import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.subheader("Available Datasets")
if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)
    st.write(f"Selected Dataset: {selected_dataset.name}, Type: {selected_dataset.type}")
    
    if selected_dataset:
        models = automl.registry.list(type="model")
        st.subheader("Available Models")
        if models:
            model_names = [model.name for model in models]
            selected_model_name = st.selectbox("Select a model", model_names)
            selected_model = next(model for model in models if model.name == selected_model_name)
            st.write(f"Selected Model: {selected_model.name}, Type: {selected_model.type}")
        else:
            st.write("No models available.")

        
else:
    st.write("No datasets available.")




