from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    A registry for managing artifacts, including saving, listing, retrieving, and deleting artifacts.
    """

    def __init__(self, database: Database, storage: Storage):
        """
        Initialize the ArtifactRegistry with a database and storage.

        :param database: The database to store artifact metadata.
        :param storage: The storage to save artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """
        Register an artifact by saving its data in storage and its metadata in the database.

        :param artifact: The artifact to register.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)
    
    def list(self, type: str = None) -> List[Artifact]:
        """
        List all artifacts, optionally filtered by type.

        :param type: The type of artifacts to list.
        :return: A list of artifacts.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts
    
    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve an artifact by its ID.

        :param artifact_id: The ID of the artifact to retrieve.
        :return: The retrieved artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )
    
    def delete(self, artifact_id: str):
        """
        Delete an artifact by its ID.

        :param artifact_id: The ID of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)
    

class AutoMLSystem:
    """
    Singleton class for managing the AutoML system, including storage and database.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initialize the AutoMLSystem with storage and database.

        :param storage: The local storage for artifacts.
        :param database: The database for artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """
        Get the singleton instance of the AutoMLSystem.

        :return: The singleton instance of AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), 
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance
    
    @property
    def registry(self):
        """
        Get the artifact registry.

        :return: The artifact registry.
        """
        return self._registry