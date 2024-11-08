from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a dataset artifact
        Args:
            *args (any): Arbituary arguments to pass to the initializer
            **kwargs (any): Arbituary keyword arguments to pass
            to the initializer
        Returns:
            None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1_0_0") -> "Dataset":
        """
        Create a Dataset from a pandas DataFrame
        Args:
            data (pd.DataFrame): DataFrame to store in the Dataset
            name (str): Name of the dataset
            asset_path (str): Path for the dataset asset
            version (str, optional): Version identifier for the dataset,
            defaults to "1_0_0"
        Returns:
            Dataset: A new Dataset containing the DataFrame data.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version
        )

    def read(self) -> pd.DataFrame:
        """
        Decode and returns the dataset data as a pandas DataFrame
        Returns:
            pd.DataFrame: DataFrame containing the decoded dataset
        """
        csv = self.data.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save a pandas DataFrame to the dataset
        Args:
            data (pd.DataFrame): DataFrame to save in the dataset
        Returns:
            bytes: The encoded CSV data as bytes
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
