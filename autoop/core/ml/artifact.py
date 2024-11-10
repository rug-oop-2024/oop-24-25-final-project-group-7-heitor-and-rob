from typing import Any, Dict, List, Optional
import base64
import pickle


class Artifact():
    def __init__(self, name: str, asset_path: str, version: str, data: bytes, type: str, tags: Optional[List[str]] = None):
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.type = type
        if tags is not None:
            self.tags = tags

    @property
    def id(self) -> str:
        encoded_path = base64.urlsafe_b64encode(
            self._asset_path.encode()).decode()
        return f"{encoded_path}:{self._version}"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "asset_path": self.asset_path,
            "version": self.version,
            "type": self.type,
            "tags": self.tags,
            "metadata": {
                "experiment_id": None,
                "run_id": None,
            }
        }

    def read(self) -> Any:
        if self.data is None:
            raise ValueError("No data found in the artifact.")
        return self.data

    def save(self, data: Any):
        with open(self.asset_path, 'wb') as f:
            pickle.dump(self, f)
