from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pandas as pd
import numpy as np

from .dataset import Dataset


class DatasetPreprocessor:
    def __init__(
        self,
        dataset: Dataset,
        split: float | None = 0.8,
        random_state: int = 0,
        stratify: bool = True,
        standardize_data: str = "minmax",
        one_hot: bool = False,
        binarize_y: bool = True,
        cross_validation_folds: int | None = None,
        fold_idx: int | None = None,
    ) -> None:
        """
        Initialize the dataset preprocessor.

        Parameters:
            - dataset: the dataset (Dataset)
            - split: the split of the dataset (float)
            - random_state: the random state (int)
            - stratify: whether to stratify the dataset (bool)
            - standardize_data: whether to standardize the dataset (str) should be one of ['minmax', 'zscore']
            - one_hot: whether to one-hot encode the dataset (bool)
            - binarize_y: whether to binarize the target variable (bool)
            - cross_validation_folds: the number of cross-validation folds (int)
            - fold_idx: the index of the fold (int)
        """
        assert (
            fold_idx is not None
            and cross_validation_folds is not None
            or fold_idx is None
            and cross_validation_folds is None
        ), "fold_idx and cross_validation_folds should be both None or both not None"

        self.dataset = dataset
        self.split = split
        self.random_state = random_state
        self.stratify = stratify
        self.standardize_data = standardize_data
        self.one_hot = one_hot
        self.binarize_y = binarize_y
        self.cross_validation_folds = cross_validation_folds
        self.fold_idx = fold_idx
        self.perform_cv = cross_validation_folds is not None and fold_idx is not None

        if self.standardize_data not in ["minmax", "zscore"]:
            raise ValueError('standardize_data should be one of ["minmax", "zscore"]')

        self.scaler = (
            StandardScaler() if self.standardize_data == "zscore" else MinMaxScaler()
        )
        self.encoder = OneHotEncoder(sparse_output=False)
        self.label_encoder = LabelEncoder()

        self.raw_df = self.dataset.raw_df
        self.target_column = self.dataset.target_column
        self.continuous_columns = self.dataset.continuous_columns
        self.categorical_columns = self.dataset.categorical_columns

        self.__initial_transform_prep()

    def __initial_transform_prep(self) -> list[pd.DataFrame]:
        """
        Prepare the initial transformation of the dataset.

        Returns:
            - the transformed dataset (list[pd.DataFrame])
        """
        X = self.raw_df.drop(columns=[self.target_column])
        y = self.raw_df[self.target_column]

        # Drop rows that contain NaN values
        X = X.dropna(how="any", axis=0)
        y = y[X.index]

        if self.perform_cv:
            if self.stratify:
                kfold = StratifiedKFold(
                    n_splits=self.cross_validation_folds,
                    random_state=self.random_state,
                    shuffle=True,
                )
            else:
                kfold = KFold(
                    n_splits=self.cross_validation_folds,
                    random_state=self.random_state,
                    shuffle=True,
                )

            fold_to_use = self.fold_idx
            for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
                if i == fold_to_use:
                    self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
                    self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
                    break

        else:
            if self.stratify:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X,
                    y,
                    test_size=1 - self.split,
                    random_state=self.random_state,
                    stratify=y,
                )
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=1 - self.split, random_state=self.random_state
                )

        # reset the index
        # X_train = X_train.reset_index(drop=True)
        # X_test = X_test.reset_index(drop=True)
        # y_train = y_train.reset_index(drop=True)
        # y_test = y_test.reset_index(drop=True)

        if self.standardize_data:
            X_train_s = self.standardize(
                self.X_train, self.continuous_columns, fit=True
            )
            X_test_s = self.standardize(self.X_test, self.continuous_columns, fit=False)

            self.X_train = self.X_train.drop(columns=self.continuous_columns)
            self.X_test = self.X_test.drop(columns=self.continuous_columns)

            self.X_train = pd.concat([self.X_train, X_train_s], axis=1)
            self.X_test = pd.concat([self.X_test, X_test_s], axis=1)

        if self.one_hot:
            X_train_o = self.one_hot_encode(
                self.X_train, self.categorical_columns, fit=True
            )
            X_test_o = self.one_hot_encode(
                self.X_test, self.categorical_columns, fit=False
            )

            # Concate only if there were categorical columns
            if self.categorical_columns:
                self.X_train = self.X_train.drop(columns=self.categorical_columns)
                self.X_test = self.X_test.drop(columns=self.categorical_columns)

                self.X_train = pd.concat([self.X_train, X_train_o], axis=1)
                self.X_test = pd.concat([self.X_test, X_test_o], axis=1)

        if self.binarize_y:
            self.y_train = self.label_encoder.fit_transform(self.y_train)
            self.y_test = self.label_encoder.transform(self.y_test)

    def one_hot_encode(
        self, X: pd.DataFrame, categorical_columns: list[str], fit: bool
    ) -> pd.DataFrame:
        """
        One-hot encode the dataset.

        Parameters:
            - X: the dataset (pd.DataFrame)
            - categorical_columns: the categorical columns (list[str])
        """
        if fit:
            self.encoder.fit(X[categorical_columns])

        X_transformed = self.encoder.transform(X[categorical_columns])
        X_transformed_features = self.encoder.get_feature_names_out(categorical_columns)
        X_transformed = pd.DataFrame(X_transformed, columns=X_transformed_features)

        self.transformed_features = X_transformed_features

        return X_transformed

    def inverse_one_hot_encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse one-hot encode the dataset.

        Parameters:
            - X: the dataset (pd.DataFrame)
        """
        X_transformed = self.encoder.inverse_transform(X)
        return X_transformed

    def standardize(
        self, X: pd.DataFrame, continuous_columns: list[str], fit: bool
    ) -> pd.DataFrame:
        """
        Standardize the dataset.

        Parameters:
            - X: the dataset (pd.DataFrame)
            - continous_columns: the continous columns (list[str])
            - fit: whether to fit the scaler (bool)
        """
        if fit:
            self.scaler.fit(X[continuous_columns])
        X_scaled = pd.DataFrame(
            self.scaler.transform(X[continuous_columns]), columns=continuous_columns
        )
        X_scaled.index = X.index
        return X_scaled

    def inverse_standardize(
        self, X: pd.DataFrame, continuous_columns: list[str]
    ) -> pd.DataFrame:
        """
        Inverse standardize the dataset.

        Parameters:
            - X: the dataset (pd.DataFrame)
            - continous_columns: the continous columns (list[str])
        """
        X_scaled = pd.DataFrame(
            self.scaler.inverse_transform(X[continuous_columns]),
            columns=continuous_columns,
        )
        return X_scaled

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get the preprocessed data.

        Returns:
            - the preprocessed data (tuple[pd.DataFrame])
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_numpy(self) -> tuple[np.ndarray]:
        """
        Get the preprocessed data in numpy format.

        Returns:
            - the preprocessed data in numpy format (tuple[np.ndarray])
        """
        X_train = self.X_train.values
        X_test = self.X_test.values
        y_train = np.array(self.y_train)
        y_test = np.array(self.y_test)
        return X_train, X_test, y_train, y_test
