from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature:
    def __init__(self, name: str, type: Literal["regression", "classification"]) -> None:
        self.name = name
        self.type = type

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"