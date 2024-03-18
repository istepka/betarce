from sklearn.model_selection import train_test_split
import yaml
import os
import logging
import pandas as pd

from create_data_examples import Dataset, DatasetPreprocessor
from mlpclassifier import MLPClassifier, train_neural_network, train_K_mlps_in_parallel

def get_config(path: str = './configv2.yml') -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(X_train, y_train, model_type: str, seed: int, hparams: dict) -> tuple:
    if model_type == 'neural_network':
        m, pp, pc = train_neural_network(X_train, y_train, seed, hparams)
    elif model_type == 'random_forest':
        raise NotImplementedError("Random forest not implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return m, pp, pc
        
def train_B(ex_type: str,
        model_type_to_use: str,
        model_base_hyperparameters: dict,
        model_fixed_hparams: dict,
        model_hyperparameters_pool: dict,
        model_fixed_seed: int,
        k_mlps_in_B: int,
        n_jobs: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> list:
    # Should bootstrap vary?
    if 'bootstrap' in ex_type.lower():
        bootstrapB = True
    else:
        bootstrapB = False
        
    # Should seed vary?
    if 'seed' in ex_type.lower():
        seedB = model_fixed_seed
    else:
        seedB = None
        
    # Should architecture vary? 
    if 'architecture' in ex_type.lower():
        fixed_hparams = False
        hparamsB = model_base_hyperparameters | model_hyperparameters_pool
    else:
        fixed_hparams = True
        hparamsB = model_base_hyperparameters | model_fixed_hparams
    
    
    if model_type_to_use == 'neural_network':
        results = train_K_mlps_in_parallel(X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hparams=hparamsB,
            bootstrap=bootstrapB,
            fixed_hparams=fixed_hparams,
            fixed_seed=seedB,
            K=k_mlps_in_B,
            n_jobs=n_jobs,
        )
        models = [model for model, _, _, _, _ in results]
    else:
        raise NotImplementedError("Random forest not implemented")
        
    return models
        

def experiment(config: dict, model_type_to_use: str = 'neural_network'):
    
    GENERAL = config['general']
    EXPERIMENTS_SETUP = config['experiments_setup']
    MODEL_HYPERPARAMETERS = config['model_hyperparameters']
    BETA_ROB = config['beta_rob']
    
    # Extract the general parameters
    cv_folds = EXPERIMENTS_SETUP['cross_validation_folds']
    global_random_state = GENERAL['random_state']
    n_jobs = GENERAL['n_jobs']
    
    # Extract the beta-robustness parameters
    k_mlps_in_B = BETA_ROB['k_mlps_in_B']
    
    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_seed']
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_hyperparameters']
    model_hyperparameters_pool = MODEL_HYPERPARAMETERS[model_type_to_use]['model_hyperparameters_pool']
    model_base_hyperparameters = MODEL_HYPERPARAMETERS[model_type_to_use]['model_base_hyperparameters']
    
    
    for ex_type in EXPERIMENTS_SETUP['ex_types']:
        logging.info(f"Running experiment type: {ex_type}")
        
        for dataset_name in EXPERIMENTS_SETUP['datasets']:
            logging.info(f"Running experiment for dataset: {dataset_name}")
            
            dataset = Dataset(dataset_name)
            
            for fold_i in range(cv_folds):
                logging.info(f"Running experiment for fold: {fold_i}")

                # Initialize the dataset preprocessor, which will handle the cross-validation and preprocessing
                dataset_preprocessor = DatasetPreprocessor(
                    dataset=dataset,
                    cross_validation_folds=cv_folds,
                    fold_idx=fold_i,
                    random_state=global_random_state,
                )
            
                # Unpack the dataset with train test from a given fold
                X_train, X_test, y_train, y_test = dataset_preprocessor.get_data()
                
                # Convert to numpy
                X_train, X_test = [x.to_numpy() for x in (X_train, X_test)]
                
                # Train M_1
                hparamsM1 = model_base_hyperparameters | model_fixed_hparams
                model1, pred_proba1, pred_crisp1 = train_model(X_train, y_train, model_type_to_use, model_fixed_seed, hparamsM1)
                
                # Train B
                modelsB = train_B(
                    ex_type=ex_type,
                    model_type_to_use=model_type_to_use,
                    model_base_hyperparameters=model_base_hyperparameters,
                    model_fixed_hparams=model_fixed_hparams,
                    model_hyperparameters_pool=model_hyperparameters_pool,
                    model_fixed_seed=model_fixed_seed,
                    k_mlps_in_B=k_mlps_in_B,
                    n_jobs=n_jobs,
                    X_train=X_train, 
                    y_train=y_train, 
                    X_test=X_test, 
                    y_test=y_test
                )
                
                    
                
                exit(0)
                


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)
    config = get_config()
    logging.debug(config)
    
    experiment(config)

