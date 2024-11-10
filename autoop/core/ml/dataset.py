from autoop.core.ml.artifact import Artifact
import base64
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class used to represent a Dataset, inheriting from Artifact.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Dataset object.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, 
                       version: str = "1.0.0") -> 'Dataset':
        """
        Create a Dataset object from a pandas DataFrame.

        :param data: The pandas DataFrame to convert.
        :param name: The name of the dataset.
        :param asset_path: The asset path for the dataset.
        :param version: The version of the dataset.
        :return: A Dataset object.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame from the stored data.

        :return: A pandas DataFrame.
        """
        data: bytes = self.data
        if isinstance(data, str):
            data = base64.b64decode(data)
        return pd.read_csv(io.BytesIO(data))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the pandas DataFrame as bytes.

        :param data: The pandas DataFrame to save.
        :return: The saved data as bytes.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
