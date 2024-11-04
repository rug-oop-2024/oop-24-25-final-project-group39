import base64
from typing import List, Dict


class Artifact():
    def __init__(self, name: str, asset_path: str, version: str,
                 data: bytes, metadata: Dict[str, str],
                 type: str, tags: List[str]) -> None:
    #def __init__(self, name: str, asset_path: str, version: str,
    #            data: bytes,
    #            type: str) -> None:
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata
        self._type = type
        self._tags = tags

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def asset_path(self):
        return self._asset_path

    @property
    def version(self):
        return self._version

    @property
    def data(self):
        return self._data

    @property
    def type(self):
        return self._type

    def generate_id(self) -> str:
        encoded_path = base64.b64encode(
            self._asset_path.endocde('utf-8')).decode('utf-8')
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """"
        returns the stored data as bytes.
        """
        return self.data

    def save(self, new_data: bytes) -> bytes:
        """
        replaces the stored data with new data.
        """
        self.data = new_data
        return self.data

    def __repr__(self) -> str:
        return (f"Artifact(id={self.id}, "
                f"asset_path={self.asset_path}, "
                f"version={self.version}, "
                f"type={self.type}, "
                f"tags={self.tags}, "
                f"metadata={self.metadata})")
