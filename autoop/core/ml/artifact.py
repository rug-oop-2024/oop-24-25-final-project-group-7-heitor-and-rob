from pydantic import BaseModel, Field, PrivateAttr
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import base64


class Artifact(BaseModel, ABC):
    _asset_path: str = PrivateAttr()
    _version: str = PrivateAttr()
    _data: bytes = PrivateAttr()
    _metadata: Dict[str, Any] = PrivateAttr()
    _type: str = PrivateAttr()
    tags: List[str] = Field(default_factory=list)

    def __init__(self, name: str, asset_path: str, version: str, data: bytes, metadata: Dict[str, Any], type: str, tags: Optional[List[str]] = None):
        super().__init__()
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata
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
            "metadata": self._metadata,
        }

    @abstractmethod
    def read(self) -> Any:
        pass

    @abstractmethod
    def save(self, data: Any):
        pass
