import base64
from typing import List, Dict
import pandas as pd
import io

class Artifact():
    # def __init__(self, name: str, asset_path: str, version: str,
    #             data: bytes, metadata: Dict[str, str],
    #            type: str, tags: List[str]) -> None:
    def __init__(self, name: str, asset_path: str, version: str,
                 data: bytes, type: str, metadata: dict = {}, tags: list = []) -> None:
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata
        self._type = type
        self._tags = tags
        self._id = self._generate_id()

    

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def asset_path(self):
        return self._asset_path
    
    @asset_path.setter
    def asset_path(self, new_asset_path):
        self._asset_path = new_asset_path

    @property
    def version(self):
        return self._version
    
    @version.setter
    def version(self, new_version):
        self._version = new_version

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data

    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, new_metadata):
        self._metadata = new_metadata

    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, new_type):
        self._type = new_type

    @property
    def tags(self):
        return self._tags
    
    @tags.setter
    def tags(self, new_tags):
        self._tags = new_tags

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, new_id):
        self._id = new_id
    
    def _generate_id(self) -> str:
        encoded_path = base64.b64encode(
            self._asset_path.encode('utf-8')).decode('utf-8')
        return f"{encoded_path}_{self.version}"

    def read(self) -> pd.DataFrame:
        """"
        returns the stored data as bytes.
        """
        df = self.data.decode()
        return pd.read_csv(io.StringIO(df))

    def save(self, new_data: bytes) -> bytes:
        """
        replaces the stored data with new data.
        """
        self.data = new_data
        return self.data

    def __repr__(self) -> str:
        return self.name
