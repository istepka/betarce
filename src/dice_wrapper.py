import dice_ml
import pandas as pd

def get_dice_explainer(
    dataset: pd.DataFrame,
    model: object,
    outcome_name: str,
    dice_method: str = 'random',
    feature_encoding: str = 'ohe-min-max',
    continous_features: list = None,
    ) -> dice_ml.Dice:
    '''
    Get a DiCE explainer.
    
    Parameters:
        - dataset: the dataset (pd.DataFrame)
        - model: the model (object)
        - outcome_name: the outcome name (str)
        - dice_method: the DiCE method (str) - 'random' or 'genetic' or 'kdtree'
        - feature_encoding: the feature encoding (str) - 'ohe-min-max' ??
        - continous_features: the continuous features (list) - if None, all features are considered continuous
    '''    
    
    if continous_features is None:
        continous_features = dataset.columns.tolist()
        
    assert outcome_name in dataset.columns, f'outcome_name must be in dataset.columns'
    
    dice_data = dice_ml.Data(
        dataframe=dataset, 
        continuous_features=continous_features,
        outcome_name=outcome_name,
        )
    
    dice_model = dice_ml.Model(model=model, backend="sklearn", func=feature_encoding)
    
    dice_exp = dice_ml.Dice(dice_data, dice_model, method=dice_method)
    
    return dice_exp

def get_dice_counterfactuals(
    dice_exp: dice_ml.Dice,
    query_instance: dict,
    total_CFs: int = 1,
    desired_class: str = 'opposite',
    proximity_weight: float = 0.2,
    diversity_weight: float = 0.01,
    classification_threshold: float = 0.5,
    random_seed: int = 123,
    ) -> pd.DataFrame:
    '''
    Get counterfactuals using DiCE.
    
    Parameters:
        - dice_exp: the DiCE explainer (dice_ml.Dice)
        - query_instance: the query instance (dict)
        - total_CFs: the number of counterfactuals (int)
        - desired_class: the desired class (str) - 'opposite' or 'random'
        - proximity_weight: the proximity weight (float)
        - diversity_weight: the diversity weight (float)
        - classification_threshold: the classification threshold (float)
        - random_seed: the random seed (int)
    '''
    
    dice_exp = dice_exp
    
    explanations_object = dice_exp.generate_counterfactuals(
        query_instances=query_instance,
        total_CFs=total_CFs,
        desired_class=desired_class,
        proximity_weight=proximity_weight,
        diversity_weight=diversity_weight,
        stopping_threshold=classification_threshold,
        random_seed=random_seed
        )
    
    explanations = explanations_object.cf_examples_list 
    
    return explanations
    


