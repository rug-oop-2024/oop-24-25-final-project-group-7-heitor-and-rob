from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import base64
import pandas as pd
import io


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"):

        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Return a pandas DataFrame from the stored data."""
        data: bytes = self.data
        if isinstance(data, str):
            data = base64.b64decode(data)
        return pd.read_csv(io.BytesIO(data))

    def save(self, data: pd.DataFrame) -> bytes:
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
