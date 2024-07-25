import numpy as np
import pandas as pd
import sklearn
import dice_ml

from .base_explainer import BaseExplainer

class DiceExplainer(BaseExplainer):
    
    def __init__(self,
                 dataset: pd.DataFrame, 
                 model: sklearn.base.BaseEstimator,
                 outcome_name: str,
                 continous_features: list = None,
        ) -> None:
        '''
        Initialize the BaseExplainer.
        
        Parameters:
            - model: the model (object)
            - outcome_name: the outcome name (str)
            - continous_features: the continuous features (list) - if None, all features are considered continuous
        '''
        self.dataset = dataset
        self.model = model
        self.outcome_name = outcome_name
        self.continous_features = continous_features
        
        self.dice_exp: dice_ml.Dice = None
        self.prep_done = False
        
    def prep(self, 
             dice_method: str = 'random',
             feature_encoding: str | None = None,
             hparams: dict = {},
        ) -> None:
        '''
        Prepare the DiCE explainer.
        
        Parameters:
            - dice_method: the DiCE method (str) - 'random' or 'genetic' or 'kdtree'
            - feature_encoding: the feature encoding (str) - 'ohe-min-max' or 'ohe' or 'binary' or 'numerical' or None
        '''
        
        if self.continous_features is None:
            self.continous_features = self.dataset.columns.tolist()
            
        assert self.outcome_name in self.dataset.columns, f'outcome_name must be in dataset.columns'
        
        dice_data = dice_ml.Data(
            dataframe=self.dataset, 
            continuous_features=self.continous_features,
            outcome_name=self.outcome_name,
            )
        
        dice_model = dice_ml.Model(model=self.model, backend="sklearn", func=feature_encoding)
        
        self.dice_exp = dice_ml.Dice(dice_data, dice_model, method=dice_method)
        
        self.prep_done = True
        
        self.hp = hparams
             
        
    def generate(self, 
            query_instance: np.ndarray | pd.DataFrame,
            total_CFs: int = 1,
            desired_class: str = 'opposite',
            proximity_weight: float = 0.2,
            diversity_weight: float = 0.05,
            classification_threshold: float = 0.5,
            random_seed: int = 123,
            sparsity_weight: float = 0.05,
        ) -> pd.DataFrame:
        '''
        Generate counterfactuals using DiCE.
        
        Parameters:
            - query_instance: the query instance (np.ndarray | pd.DataFrame)
            - total_CFs: the number of counterfactuals (int)
            - desired_class: the desired class (str) - 'opposite' or 'random'
            - proximity_weight: the proximity weight (float)
            - diversity_weight: the diversity weight (float)
            - classification_threshold: the classification threshold (float)
            - random_seed: the random seed (int)
        '''
        
        if not self.prep_done:
            raise ValueError('prep() method must be called first')
        
        if 'sparsity_weight' in self.hp:
            sparsity_weight = self.hp['sparsity_weight']
        if 'proximity_weight' in self.hp:
            proximity_weight = self.hp['proximity_weight']
        if 'diversity_weight' in self.hp:
            diversity_weight = self.hp['diversity_weight']    
        
        explanations_object = self.dice_exp.generate_counterfactuals(
            query_instances=query_instance,
            total_CFs=total_CFs,
            desired_class=desired_class,
            proximity_weight=proximity_weight,
            diversity_weight=diversity_weight,
            stopping_threshold=classification_threshold,
            sparsity_weight=sparsity_weight,
            random_seed=random_seed,
            )
        
        final_cf = explanations_object.cf_examples_list[0].final_cfs_df.to_numpy()[0][:-1]
        
        return final_cf
        