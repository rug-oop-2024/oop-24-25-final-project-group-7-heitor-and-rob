from typing import Any, Dict, List, Optional
import base64
import pandas as pd
import io


class Artifact():
    def __init__(self,
                 name: str,
                 type: str,
                 asset_path: str = "",
                 version: str = "1.0.0",
                 data: Optional[bytes] = None,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.type = type
        self.tags = [] if tags is None else tags
        self.metadata = {} if metadata is None else metadata
        self.id = f"{self.name}_{self.version}"

    def read(self) -> pd.DataFrame:
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
        self.data = base64.b64encode(data).decode("utf-8")
        return self.data
