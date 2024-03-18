import time
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os
import logging
import pandas as pd

from create_data_examples import Dataset, DatasetPreprocessor
from mlpclassifier import MLPClassifier, train_neural_network, train_K_mlps_in_parallel
from utils import bootstrap_data

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
        
def train_model_2(X_train, y_train, ex_type : str, model_type: str, hparams: dict) -> tuple:
    '''
    Train the second model, M_2, with the given hyperparameters and with proper variation
    '''
    # Should bootstrap vary?
    if 'bootstrap' in ex_type.lower():
        # If should vary, then bootstrap the data
        X_train_b, y_train_b = bootstrap_data(X_train, y_train)
    else:
        # If should not vary, then use the original data
        X_train_b, y_train_b = X_train.copy(), y_train.copy()
        
    # Should seed vary?
    if 'seed' in ex_type.lower():
        # If should not vary, then use the fixed seed
        seed = hparams['model_fixed_seed']
    else:
        # If should vary, then use a random seed
        seed = np.random.randint(0, 1000)
        
    m, pp, pc = train_model(X_train_b, y_train_b, model_type, seed, hparams)
    
    return m, pp, pc
          
def experiment(config: dict, 
        model_type_to_use: str = 'neural_network',
        save_every_n_iterations: int = 100,
    ):
    
    GENERAL = config['general']
    EXPERIMENTS_SETUP = config['experiments_setup']
    MODEL_HYPERPARAMETERS = config['model_hyperparameters']
    BETA_ROB = config['beta_rob']
    
    # Extract the results directory
    results_dir = GENERAL['result_path']
    
    # Extract the general parameters
    cv_folds = EXPERIMENTS_SETUP['cross_validation_folds']
    global_random_state = GENERAL['random_state']
    n_jobs = GENERAL['n_jobs']
    m_count_per_experiment = GENERAL['m_count_per_experiment']
    x_test_size = GENERAL['x_test_size']
    ex_types = EXPERIMENTS_SETUP['ex_types']
    datasets = EXPERIMENTS_SETUP['datasets']
    beta_confidences = EXPERIMENTS_SETUP['beta_confidences']
    delta_robustnesses = EXPERIMENTS_SETUP['delta_robustnesses'] 
    
    # Extract the beta-robustness parameters
    k_mlps_in_B = BETA_ROB['k_mlps_in_B']
    
    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_seed']
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_hyperparameters']
    model_hyperparameters_pool = MODEL_HYPERPARAMETERS[model_type_to_use]['model_hyperparameters_pool']
    model_base_hyperparameters = MODEL_HYPERPARAMETERS[model_type_to_use]['model_base_hyperparameters']
    
    
    metrics = [
        "validity", 
        "proximityL1", 
        "proximityL2",
        "plausibility", 
        "discriminative_power",
    ]
    
    results_columns = [
        'experiment_type',
        'dataset_name',
        'fold_i',
        'experiment_generalization_type',
        'x_test_sample',
        'y_test_sample',
        'model1_pred_proba',
        'model1_pred_crisp',
        'model2_name',
        'model2_pred_proba',
        'model2_pred_crisp',
        'base_counterfactual',
        'base_counterfactual_model1_pred_proba',
        'base_counterfactual_model1_pred_crisp',
        'base_counterfactual_model2_pred_proba',
        'base_counterfactual_model2_pred_crisp',
        'robust_counterfactual',
        'robust_counterfactual_model1_pred_proba',
        'robust_counterfactual_model1_pred_crisp',
        'robust_counterfactual_model2_pred_proba',
        'robust_counterfactual_model2_pred_crisp',
        'beta_confidence',
        'delta_robustness',
    ]
    
    for metric in metrics:
        results_columns.append(f"base_counterfactual_{metric}")
        results_columns.append(f"robust_counterfactual_{metric}")
    
    results_df = pd.DataFrame(columns=results_columns)
    
    global_iteration = 0
    
    for ex_type in ex_types:
        logging.info(f"Running experiment type: {ex_type}")
        
        for dataset_name in datasets:
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
                t0 = time.time()
                model1, pred_proba1, pred_crisp1 = train_model(X_train, y_train, model_type_to_use, model_fixed_seed, hparamsM1)
                time_model1 = time.time() - t0
                logging.info(f"Finished training M_1")
                
                # Train B
                t0 = time.time()
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
                time_modelsB = time.time() - t0
                logging.info(f"Finished training B")
                    
                
                for ex_generalization in ex_types:
                    
                    model2_handles = []
                    model2_times = []
                    
                    for model_2_index in range(m_count_per_experiment):
                        logging.info(f"Running experiment for generalization type: {ex_generalization}")
                        
                        # Train M_2
                        # Should architecture vary? 
                        if 'architecture' in ex_type.lower():
                            # If should vary, then sample from the pool of hyperparameters
                            hparams2 = {}
                            for _param, _options in model_hyperparameters_pool.items():
                                hparams2[_param] = np.random.choice(_options)
                            hparams2 = hparams2 | model_base_hyperparameters 
                        else:
                            # If should not vary, then use the fixed hyperparameters
                            hparams2 = model_base_hyperparameters | model_fixed_hparams
                        
                        t0 = time.time() 
                        model2, pred_proba2, pred_crisp2 = train_model_2(X_train, y_train, ex_generalization, model_type_to_use, hparams2)
                        time_model2 = time.time() - t0
                        
                        model2_handles.append((f"Model2_{model_2_index}", model2, pred_proba2, pred_crisp2))
                        model2_times.append(time_model2)
                        
                    logging.info(f"Finished training {m_count_per_experiment} M_2 models")
                    
                    
                    for x in range(x_test_size):
                        if x > X_test.shape[0] - 1:
                            logging.warning(f"Test size {x} is larger than the test set size {X_test.shape[0]}. Skipping...")
                            continue
                        
                        # Obtain the test sample
                        x_test_sample = X_test[x]
                        y_test_sample = y_test[x]
                        
                        # Obtain the predictions from M_1
                        pred_proba1_sample = pred_proba1(x_test_sample)
                        pred_crisp1_sample = pred_crisp1(x_test_sample)
                        
                        # Obtain the base counterfactual
                        
                        # Calculate metrics
                        
                        # Store in the frame
                        
                        for beta_confidence in beta_confidences:
                            for delta_robustness in delta_robustnesses:
                                for model2_name, model2, pred_proba2, pred_crisp2 in model2_handles:
                                    
                                    # Obtain the predictions from M_2
                                    pred_proba2_sample = pred_proba2(x_test_sample)
                                    pred_crisp2_sample = pred_crisp2(x_test_sample)
                                    
                                    # Obtain the robust counterfactual
                                    
                                    # Calculate the metrics
                                    
                                    # Store the results in the frame
                                    
                                    
                                    global_iteration += 1
                                    if global_iteration % save_every_n_iterations == 0:
                                        results_df.to_feather(f'./{results_dir}/{global_iteration}_results.feather')
                                    pass
                        
                        
                results_df.to_feather(f'./{results_dir}/final_results.feather')
                results_df.to_csv(f'./{results_dir}/final_results.csv')
                results_df.to_parquet(f'./{results_dir}/final_results.parquet')
                
                exit(0)
                


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)
    config = get_config()
    logging.debug(config)
    
    experiment(config)

