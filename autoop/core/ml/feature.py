from typing import Literal


class Feature():
    """Represents a feature with a name and type (regression or classification)
    """
    def __init__(self, name: str,
                 type: Literal["regression", "classification"]) -> None:
        """
        Initialize a Feature with a name and type
        Args:
            name (str): The name of the feature.
            type (Literal["regression", "classification"]): The type of
            feature, which should be either regression or classification
        """
        self.name = name
        self.type = type

    @property
    def name(self) -> str:
        """str: The name of the feature"""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set a new name for the feature
        Args:
            value (str): The new name
        Returns:
            None
        """
        if not isinstance(value, str):
            raise ValueError("Name must be a string.")
        self._name = value

    @property
    def type(self) -> Literal["regression", "classification"]:
        """Literal["regression", "classification"]: The type of feature"""
        return self._type

    @type.setter
    def type(self, new_type: Literal["regression", "classification"]) -> None:
        """
        Sets a new model type
        Args:
            new_type (Literal["regression", "classification"]): The new type,
            either regression or classification to assign to the model
        Returns:
            None
        """
        self._type = new_type

    def __str__(self) -> str:
        """
        Return a string representation of the feature
        Returns:
            str: A string representing the feature's name and type
        """
        return f"{self.name}: {self.type}"
