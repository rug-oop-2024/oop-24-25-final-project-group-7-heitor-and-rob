import pandas as pd
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects feature types in the dataset.

    Assumption: only categorical and numeric features and no NaN values.

    Args:
        dataset (Dataset): The dataset to analyze.

    Returns:
        List[Feature]: List of features with their types.

    Raises:
        ValueError: If dataset is not an instance
          of Dataset or if dataset has no data to read.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be an instance of Dataset")

    df = dataset.read()

    if df is None:
        raise ValueError(
            "Dataset has no data to read. Ensure data is initialized.")

    features = []

    for column in df.columns:
        col_data = df[column]

        if pd.api.types.is_numeric_dtype(
            col_data) or pd.api.types.is_float_dtype(
            col_data) and (col_data.max() - col_data.min() > 1):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column, type=feature_type)
        features.append(feature)

    return features
