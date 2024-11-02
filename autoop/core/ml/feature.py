
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature:
    def __init__(self, name: str, type: str) -> None:
        # Might need to change typehint for type to Literal[]?
        self.name = name
        self.type = type

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"