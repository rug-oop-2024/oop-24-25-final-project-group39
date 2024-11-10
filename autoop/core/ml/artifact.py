import base64
from typing import List, Dict
import pandas as pd
import io


class Artifact():
    """Represents an artifact with metadata, data, and
    associated properties like name, version, and tags
    """
    def __init__(self, name: str, asset_path: str,
                 data: bytes,
                 type: str,
                 version: str = "1_0_0",
                 metadata: Dict[str, str] = {},
                 tags: List[str] = []) -> None:
        """
        Initialize an Artifact instance with information and metadata
        Args:
            name (str): Name of the artifact
            asset_path (str): Path to the asset
            version (str): Version of the artifact
            data (bytes): Data associated with the artifact
            type (str): Type of artifact (e.g., file type or category)
            metadata (Dict[str, str]): Metadata dictionary for
            additional information, defaults to an empty dict
            tags (List[str], optional): Tags associated with the artifact,
            defaults to an empty list.
        Returns:
            None
        """
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata
        self._type = type
        self._tags = tags
        self._id = self._generate_id()

    @property
    def name(self) -> str:
        """Gets the name of the artifact"""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Sets the name of the artifact"""
        self._name = new_name

    @property
    def asset_path(self) -> str:
        """Gets the asset path of the artifact"""
        return self._asset_path

    @asset_path.setter
    def asset_path(self, new_asset_path: str) -> None:
        """Sets the asset path of the artifact"""
        self._asset_path = new_asset_path

    @property
    def version(self) -> str:
        """Gets the version of the artifact"""
        return self._version

    @version.setter
    def version(self, new_version: str) -> None:
        """Sets the version of the artifact"""
        self._version = new_version

    @property
    def data(self) -> bytes:
        """Gets the data associated with the artifact"""
        return self._data

    @data.setter
    def data(self, new_data: bytes) -> None:
        """Sets the data associated with the artifact"""
        self._data = new_data

    @property
    def metadata(self) -> Dict[str, str]:
        """Gets the metadata dictionary of the artifact"""
        return self._metadata

    @metadata.setter
    def metadata(self, new_metadata: Dict[str, str]) -> None:
        """Sets the metadata dictionary of the artifact"""
        self._metadata = new_metadata

    @property
    def type(self) -> str:
        """Gets the type of the artifact"""
        return self._type

    @type.setter
    def type(self, new_type: str) -> None:
        """Sets the type of the artifact"""
        self._type = new_type

    @property
    def tags(self) -> List[str]:
        """Gets the tags associated with the artifact"""
        return self._tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Sets the tags associated with the artifact"""
        self._tags = new_tags

    @property
    def id(self) -> str:
        """Gets the ID of the artifact"""
        return self._id

    @id.setter
    def id(self, new_id: str) -> None:
        """Sets the ID of the artifact"""
        self._id = new_id

    def _generate_id(self) -> str:
        """
        Generates an id for the artifact
        Returns:
            str: identifier for the artifact
        """
        encoded_path = base64.b64encode(
            self._asset_path.encode('utf-8')).decode('utf-8')
        return f"{encoded_path}_{self.version}"

    def read(self) -> pd.DataFrame:
        """
        Decodes the stored data
        Returns:
          pd.DataFrame: The stored data as bytes
        """
        df = self.data.decode()
        return pd.read_csv(io.StringIO(df))

    def save(self, new_data: bytes) -> bytes:
        """
        Replaces the stored data with new data
        Args:
            new_data (bytes): New data to save
        Returns:
            bytes: The updated data
        """
        self.data = new_data
        return self.data

    def __repr__(self) -> str:
        """
        Returns a string representation of the artifact
        Returns:
            str: Name of artifact
        """
        return self.name
