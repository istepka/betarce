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
from .CARLA.carla.recourse_methods import Face, Revise, Roar
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
        self.model_type = 'ann'
    
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


    def prep(self, method_to_use: str, hparams: dict = {}) -> None:
        '''
        Prepare the explainer.
        '''
        self.method_to_use = method_to_use
        
        match self.method_to_use:
            case 'face':
                face_hyperparams = {
                    'mode': 'knn',
                    'fraction': 0.5,
                }
                if 'fraction' in hparams:
                    face_hyperparams['fraction'] = hparams['fraction']
                if 'mode' in hparams:
                    face_hyperparams['mode'] = hparams['mode']
                    
                self.face_explainer = Face(self.model, face_hyperparams)
            case 'clue':
                # hyperparams = {
                #     "data_name": "clue_vae",
                #     "train_vae": True,
                #     "width": 128,
                #     "depth": 3,
                #     "latent_dim": 16,
                #     "batch_size": 32,
                #     "epochs": 50,
                #     "lr": 0.001,
                #     "early_stop": 5,
                # }
                # self.clue_explainer = Clue(data=self.data_catalog, mlmodel=self.model, hyperparams=hyperparams)
                hyperparams = {
                    "data_name": "revise_vae",
                    "vae_params": {
                        "layers": [23, 128],
                        "train": True,
                        "lambda_reg": 0.1,
                        "batch_size": 32,
                        "epochs": 50,
                        "lr": 0.01,
                    }
                }
                
                self.clue_explainer = Revise(self.model, self.data_catalog, hyperparams)
                
            case 'roar':
                hyperparams = {
                    'seed': 123,
                    'lime_seed': 123,
                    **hparams
                }
                self.roar_explainer = Roar(self.model, hyperparams)
            case _:
                raise ValueError(f"Method {self.method_to_use} not supported by the explainer")

    def generate(self, query_instance: pd.DataFrame) -> np.ndarray:
        if isinstance(query_instance, pd.DataFrame):
            query_instance = query_instance.copy()
            
        match self.method_to_use:
            case 'face': 
                face_cf = self.face_explainer.get_counterfactuals(query_instance)
                return face_cf.to_numpy()
            case 'clue':
                clue_cf = self.clue_explainer.get_counterfactuals(query_instance)
                return clue_cf.to_numpy()
            case 'roar':
                roar_cf = self.roar_explainer.get_counterfactuals(query_instance)
                return roar_cf.to_numpy()
            case _:
                raise ValueError(f"Method {self.method_to_use} not supported by the explainer")