from typing import List, Tuple
import pandas as pd
import json
from tqdm import tqdm
import time
import warnings
import torch
from torch import nn
import numpy as np

from .CARLA import carla
from .CARLA.carla.recourse_methods import Face
from .CARLA.carla.data.catalog import CsvCatalog
from typing import Dict, List
from .CARLA.carla import MLModel

from .base_explainer import BaseExplainer

warnings.simplefilter(action='ignore', category=FutureWarning)

class TorchModel(MLModel):

    def __init__(self, model: nn.Module, data: pd.DataFrame, columns_ohe_order: List[str]) -> None:
        super().__init__(data)
        self._mymodel = model#self.__load_model()

        self.columns_order = columns_ohe_order
    
    def __call__(self, data):
        return self._mymodel(data)

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.columns_order

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):   
        if isinstance(x, pd.DataFrame):
            x = x[self.feature_input_order].to_numpy()
            print(x)

        return self._mymodel.predict_crisp(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):

        if isinstance(x, pd.DataFrame):
            x = x[self.feature_input_order].to_numpy()

        ret = np.zeros((x.shape[0], 2))
        ret[:, 1] = self._mymodel.predict_proba(x)
        ret[:, 0] = 1 - ret[:, 1]
        return ret



class CarlaExplainer(BaseExplainer):

    def __init__(self, train_dataset: pd.DataFrame, 
        explained_model: nn.Module,
        continous_columns: List[str], 
        categorical_columns: List[str], 
        nonactionable_columns: List[str], 
        target_feature_name: str, 
        columns_order_ohe: List[str],
    ) -> None:


        self.continous_columns = continous_columns
        self.categorical_columns = categorical_columns
        self.nonactionable_columns = nonactionable_columns
        self.target_feature_name = target_feature_name
        self.data_catalog = CsvCatalog(dataset=train_dataset,
                            continuous=self.continous_columns,
                            categorical=self.categorical_columns,
                            immutables=self.nonactionable_columns,
                            target=self.target_feature_name)
        
        self.model = TorchModel(model=explained_model, data=self.data_catalog, columns_ohe_order=columns_order_ohe)


    def prep(self, method_to_use: str) -> None:
        '''
        Prepare the explainer.
        '''
        self.method_to_use = method_to_use

    def generate(self, query_instance: pd.DataFrame) -> np.ndarray:
        if isinstance(query_instance, pd.DataFrame):
            query_instance = query_instance.copy()


        if self.method_to_use == 'face':
            # Dynamically select fraction to prevent FACE from throwing errors because of the too small neighbourhood
            fraction = 0.5
            # if self.data_catalog.df_train.shape[0] * 0.05 < 50:
            #     if self.data_catalog.df_train.shape[0] * 0.1 < 50:
            #         fraction = 0.2
            #     else:
            #         fraction = 0.1
            
            face_hyperparams = {
                'mode': 'knn',
                'fraction': fraction,
            }
            self.face_explainer = Face(self.model, face_hyperparams)
            face_cf = self.face_explainer.get_counterfactuals(query_instance)
            return face_cf.to_numpy()
        else:
            raise ValueError(f"Method {self.method_to_use} not supported by the explainer")