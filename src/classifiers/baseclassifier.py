import numpy as np
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    def __init__(self, hparams: dict) -> None:
        self.hparams = hparams
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, x) -> np.ndarray[float]:
        raise NotImplementedError()

    @abstractmethod
    def predict_crisp(self, x, threshold=0.5) -> np.ndarray[int]:
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, params: dict = {}) -> None:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict[str, float]:
        raise NotImplementedError()
