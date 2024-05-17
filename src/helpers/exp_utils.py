import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import yaml


def get_config(path: str = './config.yml') -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def check_is_none(to_check: object) -> bool:
    '''Check if the object is None or np.nan or has any NaN values.'''
    if to_check is None or to_check is np.nan :
        return True
    
    if isinstance(to_check, pd.Series) or isinstance(to_check, pd.DataFrame):
        if to_check.isna().any().any():
            return True
        
    if isinstance(to_check, np.ndarray):
        if np.isnan(to_check).any():
            return True
        
    if isinstance(to_check, list):
        if pd.isna(to_check).any():
            return True
        if np.isnan(to_check).any():
            return True
        if pd.NA in to_check:
            return True
    
    return False
    
def sample_architectures(n: int, hparams: dict) -> list[dict]:
    '''
    Sample n architectures from the hyperparameters pool
    '''
    architectures = []
    for _ in range(n):
        architecture = {}
        for _param, _options in hparams.items():
            if isinstance(_options, list):
                architecture[_param] = np.random.choice(_options)
            elif isinstance(_options, dict):
                lower = _options['lower']
                upper = _options['upper']
                # Check if the lower and upper are integers
                if isinstance(lower, int) and isinstance(upper, int):
                    lower = int(lower)
                    upper = int(upper)
                    architecture[_param] = np.random.randint(lower, upper + 1)
                # Otherwise, they are floats
                else:
                    freq = _options['freq']
                    lower, upper, freq = float(lower), float(upper), int(freq)
                    architecture[_param] = np.random.uniform(lower, upper, freq)
            else:
                raise ValueError('Unknown hyperparameter type', _options, 'for', _param)
        architectures.append(architecture)
    return architectures

def sample_seeds(n: int) -> list[int]:
    '''
    Sample n seeds for the bootstrap
    '''
    seeds = np.random.choice(1000, n, replace=False)
    return seeds

def calculate_metrics(cf: np.ndarray, 
        cf_desired_class: int,
        x: np.ndarray, 
        y_train: np.ndarray,
        nearest_neighbors_model: NearestNeighbors,
        predict_fn_crisp: callable,
        dpow_neighbours: int = 15,
        plausibility_neighbours: int = 15,
    ) -> dict[str, float | int]:
    '''
    Calculates the metrics for a counterfactual example.
    '''
    
    if check_is_none(cf):
        return {
            'validity': np.nan,
            'proximityL1': np.nan,
            'proximityL2': np.nan,
            'dpow': np.nan,
            'plausibility': np.nan
        }
    
    cf_label = predict_fn_crisp(cf)[0]
    
    # Validity
    validity = int(int(cf_label) == cf_desired_class)
    
    # Proximity L1
    proximityL1 = np.sum(np.abs(x - cf))
    
    # Proximity L2
    proximityL2 = np.sqrt(np.sum(np.square(x - cf)))
    
    # Discriminative Power (fraction of neighbors with the same label as the counterfactual)
    neigh_indices = nearest_neighbors_model.kneighbors(cf.reshape(1, -1), return_distance=False, n_neighbors=dpow_neighbours)
    neigh_labels = y_train[neigh_indices[0]]
    dpow = np.sum(neigh_labels == cf_label) / len(neigh_labels) # The fraction of neighbors with the same label as the counterfactual
    
    # Plausibility (average distance to the 50 nearest neighbors in the training data)
    neigh_dist, _ = nearest_neighbors_model.kneighbors(cf.reshape(1, -1), return_distance=True, n_neighbors=plausibility_neighbours)
    plausibility = np.mean(neigh_dist[0])
    
    return {
        'validity': validity,
        'proximityL1': proximityL1,
        'proximityL2': proximityL2,
        'dpow': dpow,
        'plausibility': plausibility
    }

