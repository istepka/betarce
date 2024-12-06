import pandas as pd
from abc import ABC, abstractmethod


class PostHocExplainer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def prep(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, x) -> pd.DataFrame:
        raise NotImplementedError
