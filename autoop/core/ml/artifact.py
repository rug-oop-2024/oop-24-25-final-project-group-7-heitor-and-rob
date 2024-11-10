from typing import Any, Dict, List, Optional
import base64
import pandas as pd
import io


class Artifact:
    """
    A class to represent an artifact.

    Attributes
    ----------
    name : str
        The name of the artifact.
    type : str
        The type of the artifact.
    asset_path : str
        The path to the asset.
    version : str
        The version of the artifact.
    data : Optional[bytes]
        The data of the artifact.
    tags : Optional[List[str]]
        The tags associated with the artifact.
    metadata : Optional[Dict[str, Any]]
        The metadata of the artifact.
    id : str
        The unique identifier of the artifact.
    """

    def __init__(self,
                 name: str,
                 type: str,
                 asset_path: str = "",
                 version: str = "1.0.0",
                 data: Optional[bytes] = None,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Constructs all the necessary attributes for the artifact object.

        Parameters
        ----------
        name : str
            The name of the artifact.
        type : str
            The type of the artifact.
        asset_path : str, optional
            The path to the asset (default is "").
        version : str, optional
            The version of the artifact (default is "1.0.0").
        data : Optional[bytes], optional
            The data of the artifact (default is None).
        tags : Optional[List[str]], optional
            The tags associated with the artifact (default is None).
        metadata : Optional[Dict[str, Any]], optional
            The metadata of the artifact (default is None).
        """
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.type = type
        self.tags = [] if tags is None else tags
        self.metadata = {} if metadata is None else metadata
        self.id = f"{self.name}_{self.version}"

    def read(self) -> pd.DataFrame:
        """
        Reads the data of the artifact and returns it as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The data of the artifact as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the data is not a string or bytes.
        """
        if isinstance(self.data, str):
            self.data = self.data.encode()

        if isinstance(self.data, bytes):
            try:
                decoded = base64.b64decode(self.data)
                return pd.read_csv(io.BytesIO(decoded))
            except Exception:
                return pd.read_csv(io.BytesIO(self.data))
        else:
            raise ValueError("Data is not a string or bytes.")

    def save(self, data: Any):
        """
        Saves the given data to the artifact.

        Parameters
        ----------
        data : Any
            The data to be saved.

        Returns
        -------
        str
            The base64 encoded string of the saved data.
        """
        self.data = base64.b64encode(data).decode("utf-8")
        return self.data
