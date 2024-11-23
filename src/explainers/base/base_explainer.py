import pandas as pd
from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    def __init__(
        self,
        dataset: pd.DataFrame,
        model: object,
        outcome_name: str,
        continous_features: list = None,
    ) -> None:
        """
        Initialize the BaseExplainer.

        Parameters:
            - model: the model (object)
            - outcome_name: the outcome name (str)
            - continous_features: the continuous features (list) - if None, all features are considered continuous
        """
        self.dataset = dataset
        self.model = model
        self.outcome_name = outcome_name
        self.continous_features = continous_features

    @abstractmethod
    def prep(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, x) -> pd.DataFrame:
        raise NotImplementedError
