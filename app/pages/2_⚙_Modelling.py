import streamlit as st
import pandas as pd
import os
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import CLASSIFICATION_MODELS, REGRESSION_MODELS
from autoop.core.ml.model import get_model
from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.artifact import Artifact

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Write helper text in a styled format.

    :param text: The text to display as helper text.
    :type text: str
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    """In this section, you can design a
      machine learning pipeline to train a model on a dataset."""
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.subheader("Available Datasets")
write_helper_text(
    """Choose a dataset to use for modelling.
      The dataset will be used to train the """
    "model."
)
if datasets:
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(
        dataset for dataset in datasets if
        dataset.name == selected_dataset_name
    )
    st.write(
        f"Selected Dataset: {selected_dataset.name}, Type: {selected_dataset.type}")

    if selected_dataset:
        st.write(f"### Selected Dataset: {selected_dataset.name}")
        st.write(f"Type: {selected_dataset.type}")
        try:
            data = selected_dataset.read()
            st.write("### Data Preview")
            st.dataframe(data.head())

            features = detect_feature_types(selected_dataset)
            feature_names = [feature.name for feature in features]

            st.subheader("Feature Detection")
            write_helper_text(
                """Select the input features and
                  the target feature. The target feature """
                """will be used to detect the task type (
                classification or regression)."""
            )
            input_features = st.multiselect(
                "Select Input Features", feature_names
            )
            available_target_features = [
                feature for feature in feature_names if feature not in input_features]
            target_feature = st.selectbox(
                "Select Target Feature", available_target_features
            )

            target_feature_type = next(
                feature.type for feature in features if feature.name == target_feature)

            if target_feature_type == "numerical":
                task_type = "regression"
                available_models = REGRESSION_MODELS
                available_metrics = METRICS[:3]
            elif target_feature_type == "categorical":
                task_type = "classification"
                available_models = CLASSIFICATION_MODELS
                available_metrics = METRICS[3:]

            st.write(f"### Detected Task Type: {task_type}")

            st.subheader("Model Selection")
            write_helper_text("Select a model based on the task type."
                              )
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
                "Available Metrics", available_metrics)
            write_helper_text(
                """Choose the metric(s)
                to evaluate the model's performance."""
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
            input = [
                feature for feature in features if feature.name in input_features]
            target = next(
                feature for feature in features if feature.name == target_feature)

            pipeline = Pipeline(
                model=model,
                metrics=metrics,
                dataset=selected_dataset,
                input_features=input,
                target_feature=target,
                split=split_ratio
            )

            if st.button("Train Model"):
                st.write("Training model...")
                try:
                    results = pipeline.execute()
                    st.success("Model trained successfully!")

                    write_helper_text(
                        """Press the view results button below
                            to see more details."""
                    )

                    with st.expander("View Results"):
                        st.json(pipeline._evaluate())

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

            st.subheader("Save Pipeline")
            write_helper_text(
                "Provide a name and version for your pipeline to save it as an artifact.")

            pipeline_name = st.text_input("Enter Pipeline Name", "my_pipeline")
            pipeline_version = st.text_input("Enter Pipeline Version", "1.0.0")

            st.write("Debug: Current working directory:", os.getcwd())

            if pipeline_name and pipeline_version:
                asset_path = f"pipelines/{pipeline_name}_{pipeline_version}.pkl"

                if st.button("Save Pipeline as Artifact"):
                    st.write("Saving pipeline as artifact...")
                    try:
                        pipeline_data = {
                            "model": pipeline._model,
                            "input_features": pipeline._input_features,
                            "target_feature": pipeline._target_feature,
                            "split": pipeline._split,
                            "artifacts": pipeline._artifacts,
                            "metrics": pipeline._metrics,
                        }
                        serialized_pipeline = pickle.dumps(pipeline_data)

                        new_pipeline_artifact = Artifact(
                            name=pipeline_name,
                            version=pipeline_version,
                            type="pipeline",
                            asset_path=asset_path,
                            data=serialized_pipeline,
                            tags=["pipeline", "automl"],
                            metadata={
                                "input_features": [feature.name for feature in pipeline._input_features],
                                "target_feature": pipeline._target_feature.name,
                                "model": pipeline._model.type,
                                "task_type": "regression" if pipeline._target_feature.type == "continuous" else "classification",
                                "split_ratio": pipeline._split,
                                "metrics": [str(metric) for metric in pipeline._metrics]
                            }
                        )
                        automl.registry.register(new_pipeline_artifact)
                        st.success(
                            f"Pipeline '{pipeline_name}' version '{pipeline_version}' saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving pipeline: {str(e)}")

        except FileNotFoundError:
            st.error("Dataset file not found.")
        except Exception as e:
            st.error(
                f"Error reading dataset: {str(e)}"
            )
    else:
        st.write("No datasets available.")
