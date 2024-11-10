from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found
    """
    def __init__(self, path: str) -> None:
        """
        Raises exception when a specified path is not found
        Args:
            path (str): The path that was not found'
        Returns:
            None
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage operations
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        Returns:
            None
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            List[str]: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    A concrete implementation of the Storage class that saves, loads,
    deletes, and lists files locally
    """
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the LocalStorage with a base path
        Args:
            base_path (str): Base directory for storage,
            defaults to "./assets"
        Returns:
            None
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a given key
        Args:
            data (bytes): Data to save
            key (str): Key (path relative to base path) to save data at
        Returns:
            None
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a given key
        Args:
            key (str): Key to load data
        Returns:
            bytes: Loaded data
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = '\\') -> None:
        r"""
        Delete data at a given key
        Args:
            key (str): Key to delete data,
                defaults to '\\' for Windows
        Returns:
            None
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = '\\') -> List[str]:
        r"""
        List all paths under a given prefix
        Args:
            prefix (str): Prefix to list files,
                defaults to '\\' for Windows
        Returns:
            List[str]: List of paths
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in
                keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a path exists, raising an error if not
        Args:
            path (str): The path to check
        Raises:
            NotFoundError: If the path does not exist
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with a relative path
        Args:
            path (str): The path to join with the base path
        Returns:
            str: The full path with the base path and the given path
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
