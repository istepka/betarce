import numpy as np
from torch import nn


class BaseClassifier():
    def __init__(self, hparams: dict) -> None:
        self.hparams = hparams
        self.model = self.build_model()
        
    def build_model(self):
        raise NotImplementedError()
        
    def forward(self, x):
        raise NotImplementedError()
    
    def predict_proba(self, x) -> np.ndarray[float]:
        raise NotImplementedError()
    
    def predict_crisp(self, x, threshold=0.5) -> np.ndarray[int]:
        raise NotImplementedError()
    
    def fit(self, X_train, y_train, X_val, y_val, params: dict = {}) -> None:
        raise NotImplementedError()
                                
    def evaluate(self, X_test, y_test) -> dict[str, float]:
        raise NotImplementedError()
        
     
    