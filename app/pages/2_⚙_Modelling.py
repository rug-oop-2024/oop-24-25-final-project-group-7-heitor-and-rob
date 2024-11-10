import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import CLASSIFICATION_MODELS, REGRESSION_MODELS


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.subheader("Available Datasets")
if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(
        dataset for dataset in datasets if dataset.name == selected_dataset_name)
    st.write(
        f"Selected Dataset: {selected_dataset.name}, Type: {selected_dataset.type}")

    if selected_dataset:
        st.write(f"### Selected Dataset: {selected_dataset.name}")
        st.write(f"Type: {selected_dataset.type}")
        try:
            data = selected_dataset.read()
            st.write("### Data Preview")
            st.dataframe(data.head())

            st.subheader("Feature Detection")
            feature_columns = data.columns.tolist()
            st.write("Select input features and a target feature from the dataset:")
            input_features = st.multiselect(
                "Select Input Features", feature_columns)
            target_feature = st.selectbox(
                "Select Target Feature", feature_columns)

            if input_features and target_feature:
                if target_feature in input_features:
                    st.error(
                        "Target feature cannot be selected as input feature.")
                else:
                    if pd.api.types.is_numeric_dtype(data[target_feature]) or pd.api.types.is_float_dtype(data[target_feature]) and (data[target_feature].max() - data[target_feature].min() > 1):
                        type = "numerical"
                        available_models = REGRESSION_MODELS
                    else:
                        type = "categorical"
                        available_models = CLASSIFICATION_MODELS
                    st.write(f"### Detected task type: {type}")
                    st.info(
                        f"Task type based on the target feature '{target_feature}': {type}")

                    st.subheader("Select a model")
                    selected_model = st.selectbox(
                        "Select a model", available_models)

                    if selected_model:
                        st.write(
                            f"Selected model: {selected_model}")
                    else:
                        st.write("No models available.")

        except FileNotFoundError:
            st.error("Dataset file not found.")
        except Exception as e:
            st.error(f"Error reading dataset: {str(e)}")
    else:
        st.write("No datasets available.")
