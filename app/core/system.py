from autoop.core.storage import LocalStorage
from autoop.core.database import Database
#  from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """
    A class that manages the registration, retrieval, and deletion of artifacts
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initializes the ArtifactRegistry
        Args:
            database (Database): The database for storing artifact metadata
            storage (Storage): The storage for saving artifact files
        Returns:
            None
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving its data in storage and
        metadata in the database
        Args:
            artifact (Artifact): The artifact to register
        Returns:
            None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
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
        Retrieves a list of artifacts
        Args:
            type (str): The type of artifacts to list
        Returns:
            List[Artifact]: A list of Artifact objects.
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
        Retrieves an artifact
        Args:
            artifact_id (str): The ID of the artifact
        Returns:
            Artifact: The retrieved artifact
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

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact
        Args:
            artifact_id (str): The ID of the artifact
        Returns:
            None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem():
    """
    A singleton class that manages the entire AutoML system, including storage,
    database management, and artifact registration
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the AutoMLSystem
        Args:
            storage (LocalStorage): The storage for managing artifact files
            database (Database): The database for managing artifact metadata
        Returns:
            None
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Retrieves the singleton instance of the AutoMLSystem
        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem
        """
        # Check about the backslashes, maybe change this to generic
        # for all operating systems
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage(".\\assets\\objects"),
                Database(
                    LocalStorage(".\\assets\\dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Provides access to the artifact registery
        Returns:
            ArtifactRegistry: The registry for managing artifacts
        """
        return self._registry