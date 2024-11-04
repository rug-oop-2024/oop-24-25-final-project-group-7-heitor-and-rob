
import pandas as pd
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
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

        # here we're checking if the values in the column are numeric and if the range of the values in that column is wider than 1 (indication of continuous data)
        if pd.api.types.is_numeric_dtype(col_data) and (col_data.max() - col_data.min() > 1):
            feature_type = "numeric"
        else:
            feature_type = "categorical"

        feature = Feature(name=column, type=feature_type)
        features.append(feature)

    return features
