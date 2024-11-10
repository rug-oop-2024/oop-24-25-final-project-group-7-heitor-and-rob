import streamlit as st
import pandas as pd
import os
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import CLASSIFICATION_MODELS, REGRESSION_MODELS, get_model
from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """
    Write helper text in a styled format.

    :param text: The text to display as helper text.
    :type text: str
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.subheader("Available Datasets")
write_helper_text(
    "Choose a dataset to use for modelling. The dataset will be used to train the model."
)
if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(
        dataset for dataset in datasets if dataset.name == selected_dataset_name
    )
    st.write(
        f"Selected Dataset: {selected_dataset.name}, Type: {selected_dataset.type}"
    )

    if selected_dataset:
        st.write(f"### Selected Dataset: {selected_dataset.name}")
        st.write(f"Type: {selected_dataset.type}")
        try:
            data = selected_dataset.read()
            st.write("### Data Preview")
            st.dataframe(data.head())

            st.subheader("Feature Detection")
            write_helper_text(
                """Select the input features and the target feature.
                  The target feature will be used to detect the task type (classification or regression)."""
            )
            feature_columns = data.columns.tolist()
            input_features = st.multiselect(
                "Select Input Features", feature_columns
            )
            target_feature = st.selectbox(
                "Select Target Feature", feature_columns
            )

            if input_features and target_feature:
                if target_feature in input_features:
                    st.error(
                        "Target feature cannot be selected as input feature."
                    )
                else:
                    if pd.api.types.is_numeric_dtype(data[target_feature]
                                                     ) or pd.api.types.is_float_dtype(
                                                         data[target_feature]) and (
                                                             data[target_feature].max(

                                                             ) - data[target_feature].min(

                                                             ) > 1):
                        type = "numerical"
                        available_models = REGRESSION_MODELS
                        available_metrics = METRICS[:3]
                    else:
                        type = "categorical"
                        available_models = CLASSIFICATION_MODELS
                        available_metrics = METRICS[3:]
                    st.write(f"### Detected task type: {type}")
                    st.info(
                        f"Task type based on the target feature '{target_feature}': {type}"
                    )

                    st.subheader("Model Selection")
                    write_helper_text("Select a model based on the task type.")
                    selected_model = st.selectbox(
                        "Select a model", available_models
                    )

                    if selected_model:
                        st.write(
                            f"Selected model: {selected_model}"
                        )
                    else:
                        st.write("No models available.")

                    st.subheader("Select Dataset Split")
                    write_helper_text(
                        "Choose a split ratio for training and testing data."
                    )
                    split_ratio = st.slider(
                        "Training/Test Data Split Ratio",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                    )

                    st.subheader("Select Metrics")
                    metrics = st.multiselect(
                        "Available Metrics", available_metrics
                    )
                    write_helper_text(
                        "Choose the metric(s) to evaluate the model's performance."
                    )

                    st.subheader("Pipeline Summary")
                    write_helper_text(
                        "Make sure pipeline configurations are correct."
                    )
                    st.markdown(
                        f"""
                        - Dataset: {selected_dataset.name}
                        - Input Features: {', '.join(input_features)}
                        - Target Feature: {target_feature}
                        - Model: {selected_model}
                        - Split Ratio: {split_ratio}
                        - Metrics: {', '.join(metrics)}
                        """
                    )
                    # this is where it stops working and i dont know why
                    model = get_model(selected_model)
                    metrics = [get_metric(metric) for metric in metrics]
                    pipeline = Pipeline(
                        model=model,
                        metrics=metrics,
                        dataset=selected_dataset,
                        input_features=input_features,
                        target_feature=target_feature,
                        split=split_ratio
                    )

                    if st.button("Train Model"):
                        st.write("Training model...")
                        try:
                            results = pipeline.execute()
                            st.success("Model trained successfully!")

                            write_helper_text(
                                "Press the view results button below to see more details.")

                            with st.expander("View Results"):
                                st.json(pipeline._evaluate())

                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")

                    st.subheader("Save Pipeline")
                    write_helper_text(
                        "Save the pipeline to the registry for future use.")
                    st.write(
                        "The saved pipeline will be trained regardless of whether you pressed the train model button above.")

                    pipeline_name = st.text_input("Pipeline Name")
                    if st.button("Save Pipeline"):
                        st.write("Saving pipeline...")
                        results = pipeline.execute()
                        pipeline_dir = "assets/pipelines"
                        if not os.path.exists(pipeline_dir):
                            os.makedirs(pipeline_dir)

                        pipeline_path = os.path.join(
                            pipeline_dir, f"{pipeline_name}.pkl")
                        with open(pipeline_path, 'wb') as f:
                            try:
                                automl.registry.save(pipeline)
                                st.success("Pipeline saved successfully!")
                            except Exception as e:
                                st.error(f"Error saving pipeline: {str(e)}")

        except FileNotFoundError:
            st.error("Dataset file not found.")
        except Exception as e:
            st.error(
                f"Error reading dataset: {str(e)}, the problem is with {e}")
    else:
        st.write("No datasets available.")
