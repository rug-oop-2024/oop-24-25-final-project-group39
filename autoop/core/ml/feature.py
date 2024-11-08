from typing import Literal


class Feature():
    # attributes here
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

    def __str__(self) -> str:
        """
        Return a string representation of the feature
        Returns:
            str: A string representing the feature's name and type
        """
        return f"{self.name}: {self.type}"
