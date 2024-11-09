from typing import Any, Dict, List, Optional
import base64
import pickle


class Artifact(BaseModel, ABC):
    name: str = Field()
    _asset_path: str = PrivateAttr()
    _version: str = PrivateAttr()
    _data: bytes = PrivateAttr()
    _type: str = PrivateAttr()
    tags: List[str] = Field(default_factory=list)

    def __init__(self, name: str, asset_path: str, version: str, data: bytes, type: str, tags: Optional[List[str]] = None):
        super().__init__(name=name, tags=tags or [])
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._type = type
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
            "asset_path": self._asset_path,
            "version": self._version,
            "type": self._type,
            "tags": self.tags,
            "metadata": {
                "experiment_id": None,
                "run_id": None,
            }
        }

    @abstractmethod
    def read(self) -> Any:
        if self._data is None:
            raise ValueError("No data found in the artifact.")
        return self._data

    @abstractmethod
    def save(self, data: Any):
        with open(self._asset_path, 'wb') as f:
            pickle.dump(self, f)
