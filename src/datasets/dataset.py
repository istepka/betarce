import pandas as pd
import numpy as np
from .dataset_wrappers import *  # noqa


class Dataset:
    def __init__(self, name: str, random_state: int = 0) -> None:
        """
        Initialize the dataset.

        Parameters:
            - name: the name of the dataset (str) should be one of ['german', 'fico', 'compas', 'donuts', 'moons', 'breast_cancer', 'wine_quality', 'diabetes', 'german_binary']
            - random_state: the random state (int)
        """

        self.name = name
        self.random_state = random_state

        try:
            self.data = globals().get(self.name)()
        except Exception as e:
            raise ValueError(f"Dataset {self.name} is not available. Exception: {e}")

        self.raw_df = self.data["raw_df"]
        self.target_column = self.data["target_column"]
        self.continuous_columns = self.data["continuous_columns"]
        self.categorical_columns = self.data["categorical_columns"]
        self.freeze_columns = self.data["freeze_columns"]
        self.feature_ranges = self.data["feature_ranges"]

    def __str__(self) -> str:
        return f"Dataset_{self.name}"

    def __repr__(self) -> str:
        return f"Dataset_{self.name}"

    def __len__(self) -> int:
        return len(self.raw_df)

    def __getitem__(self, index: int) -> pd.Series:
        return self.raw_df.iloc[index]

    def get_numpy(self) -> tuple[np.ndarray]:
        """
        Get the dataset in numpy format.

        Returns:
            - the dataset in numpy format (tuple[np.ndarray])
        """
        X = self.raw_df.drop(columns=[self.target_column]).values
        y = self.raw_df[self.target_column].values
        return X, y

    def get_raw_df(self) -> pd.DataFrame:
        """
        Get the raw dataframe.

        Returns:
            - the raw dataframe (pd.DataFrame)
        """
        return self.raw_df

    def overwrite_raw_df(self, new_raw_df: pd.DataFrame) -> None:
        """
        Overwrite the raw dataframe.

        Parameters:
            - new_raw_df: the new raw dataframe (pd.DataFrame)
        """
        self.raw_df = new_raw_df
        self.data["raw_df"] = new_raw_df

    def get_target_column(self) -> str:
        """
        Get the target column.

        Returns:
            - the target column (str)
        """
        return self.target_column

    def get_continuous_columns(self) -> list[str]:
        """
        Get the continuous columns.

        Returns:
            - the continuous columns (list[str])
        """
        return self.continuous_columns

    def get_categorical_columns(self) -> list[str]:
        """
        Get the categorical columns.

        Returns:
            - the categorical columns (list[str])
        """
        return self.categorical_columns

    def get_original_features(self) -> list[str]:
        """
        Get the original features.

        Returns:
            - the original features (list[str])
        """
        return self.raw_df.columns.tolist()
