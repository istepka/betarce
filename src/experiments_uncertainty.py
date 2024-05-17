import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy import stats
import yaml
import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from helpers.data_handler import Dataset, DatasetPreprocessor
from models.mlpclassifier import MLPClassifier, train_neural_network
from models.rfclassifier import RFClassifier, train_random_forest
from models.dtclassifier import DecisionTree, train_decision_tree
from models.lgbmclassifier import LGBMClassifier, train_lgbm
from models.baseclassifier import BaseClassifier
from models.utils import bootstrap_data

from helpers.exp_utils import check_is_none, sample_architectures, sample_seeds, calculate_metrics, get_config

from uncertainty.EDL import EDLModel, EDL, edl_digamma_loss, edl_mse_loss, edl_log_loss

from explainers import DiceExplainer, GrowingSpheresExplainer, BaseExplainer
from robx import robx_algorithm
from uncertain_robx import uncertain_robx_algorithm

def train_EDL(X_train, y_train, seed: int, hparams: dict) -> EDLModel:
    
    # Transform y_train to one-hot
    y_train = np.eye(2)[y_train.astype(int)]
    
    _X_train, _X_test, _y_train, _y_test = train_test_split(X_train, y_train, test_size=hparams['test_split_frac'], random_state=42)
    
    _X_train = torch.tensor(_X_train, dtype=torch.float32)
    _y_train = torch.tensor(_y_train, dtype=torch.float32)
    _X_test = torch.tensor(_X_test, dtype=torch.float32)
    _y_test = torch.tensor(_y_test, dtype=torch.float32)
    
    best_val_loss = np.inf
    best_model = None
    
    for i in range(hparams['select_best']):
        model = EDLModel(input_size=X_train.shape[1])
        EDLTrainer = EDL(model, _X_train, _y_train, _X_test, _y_test, criterion=hparams['criterion'], lr=hparams['lr'], batch_size=hparams['batch_size'], seed=seed+i)
        EDLTrainer.train(epochs=hparams['epochs'], verbose=True, early_stopping=True)
        
        if EDLTrainer.min_val_loss < best_val_loss:
            best_val_loss = EDLTrainer.min_val_loss
            best_model = model
            
    best_model.eval()
            
    return best_model
            

def train_model(X_train, y_train, model_type: str, seed: int, hparams: dict) -> tuple:
    if model_type == 'neural_network':
        m, pp, pc = train_neural_network(X_train, y_train, seed, hparams)
    elif model_type == 'random_forest':
        m, pp, pc = train_random_forest(X_train, y_train, seed, hparams)
    elif model_type == 'decision_tree':
        m, pp, pc = train_decision_tree(X_train, y_train, seed, hparams)
    elif model_type == 'lightgbm':
        m, pp, pc = train_lgbm(X_train, y_train, seed, hparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return m, pp, pc
       
def train_model_2(X_train, y_train,
        perturbation_types: str,
        model_type: str,
        hparams: dict,
        seed: int,
        bootstrap_seed: int,
    ) -> tuple:
    
    # Perform bootrap with the bootstrap seed
    if 'bootstrap' in perturbation_types.lower():
        X_train_b, y_train_b = bootstrap_data(X_train, y_train, bootstrap_seed)
    else:
        X_train_b, y_train_b = X_train, y_train
        
    print(f"Training model 2 with seed: {seed}, {bootstrap_seed}, {hparams}")
        
    m, pp, pc = train_model(X_train_b, y_train_b, model_type, seed, hparams)
    
    return m, pp, pc
          
def prepare_base_counterfactual_explainer(
        base_cf_method: str,
        hparams: dict,
        model: BaseClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        dataset_preprocessor: DatasetPreprocessor,
        predict_fn_1_crisp: callable,  
    ) -> BaseExplainer:
    
    
    match base_cf_method:
        case 'dice':
            X_train_w_target = X_train.copy()
            X_train_w_target[dataset_preprocessor.target_column] = y_train
            explainer = DiceExplainer(
                dataset=X_train_w_target,
                model=model,
                outcome_name=dataset_preprocessor.target_column,
                continous_features=dataset_preprocessor.continuous_columns
            )
            explainer.prep(
                dice_method='random',
                feature_encoding=None
            )
        case 'gs':
        
            explainer = GrowingSpheresExplainer(
                keys_mutable=dataset_preprocessor.X_train.columns.tolist(),
                keys_immutable=[],
                feature_order=dataset_preprocessor.X_train.columns.tolist(),
                binary_cols=dataset_preprocessor.transformed_features.tolist(),
                continous_cols=dataset_preprocessor.continuous_columns,
                pred_fn_crisp=predict_fn_1_crisp,
                target_proba=hparams['target_proba'],
                max_iter=hparams['max_iter'],
                n_search_samples=hparams['n_search_samples'],
                p_norm=hparams['p_norm'],
                step=hparams['step']
            )    
            explainer.prep()
        case _:
            raise ValueError('base_cf_method must be either "dice" or "gs"')
        
    return explainer

def base_counterfactual_generate(
    base_explainer: BaseExplainer,
    instance: pd.DataFrame,
    **kwargs,
    ) -> np.ndarray:
    
    if isinstance(base_explainer, DiceExplainer):
        return base_explainer.generate(instance, **kwargs)
    elif isinstance(base_explainer, GrowingSpheresExplainer):
        return base_explainer.generate(instance)
    else:
        raise ValueError('base_explainer must be either a DiceExplainer or a GrowingSpheresExplainer')
          
def experiment(config: dict):
    
    GENERAL = config['general']
    EXPERIMENTS_SETUP = config['experiments_setup']
    MODEL_HYPERPARAMETERS = config['model_hyperparameters']
    ROBUST_CF_METHODS = config['robust_cf_methods']
    CF_METHODS = config['cf_methods']
    
    # Extract the results directory
    results_dir = GENERAL['result_path']
    results_df_dir = os.path.join(results_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_df_dir, exist_ok=True)
    
    # Save the config right away
    with open(os.path.join(results_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)
    
    # Extract the general parameters
    global_random_state = GENERAL['random_state']
    n_jobs = GENERAL['n_jobs']
    save_every_n_iterations = GENERAL['save_every_n_iterations']
    
    robust_cf_method = EXPERIMENTS_SETUP['robust_cf_method']
    m_count_per_experiment = EXPERIMENTS_SETUP['m_count_per_experiment']
    x_test_size = EXPERIMENTS_SETUP['x_test_size']
    perturbation_types = EXPERIMENTS_SETUP['perturbation_types']
    datasets = EXPERIMENTS_SETUP['datasets']
    model_type_to_use = EXPERIMENTS_SETUP['model_type_to_use']
    base_cf_method = EXPERIMENTS_SETUP['base_counterfactual_method']
    
    
    cv_folds = EXPERIMENTS_SETUP['cross_validation_folds']
    train_test_split = EXPERIMENTS_SETUP['train_test_split']

    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_seed']
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_hyperparameters']
    model_hyperparameters_pool = MODEL_HYPERPARAMETERS[model_type_to_use]['model_hyperparameters_pool']
    model_base_hyperparameters = MODEL_HYPERPARAMETERS[model_type_to_use]['model_base_hyperparameters']
   
    results_df = pd.DataFrame()
    
    global_iteration = 0
    all_iterations =  len(datasets) * cv_folds * m_count_per_experiment  * x_test_size
    
    if robust_cf_method == 'robx': # If robx is used, then the iterations are multiplied by the number of taus
        all_iterations *= len(ROBUST_CF_METHODS['robx']['taus']) * len(ROBUST_CF_METHODS['robx']['variances'])
        
    tqdm_pbar = tqdm(total=all_iterations, desc="Overall progress")

    for dataset_name in datasets:
        logging.info(f"Running experiment for dataset: {dataset_name}")
        
        dataset = Dataset(dataset_name)
        
        for fold_i in range(cv_folds):
            logging.info(f"Running experiment for fold: {fold_i}")

            # Initialize the dataset preprocessor, which will handle the cross-validation and preprocessing
            if cv_folds == 1:
                dataset_preprocessor = DatasetPreprocessor(
                    dataset=dataset,
                    split=train_test_split,
                    random_state=global_random_state,
                    one_hot=True,
                )
            else:
                dataset_preprocessor = DatasetPreprocessor(
                    dataset=dataset,
                    cross_validation_folds=cv_folds,
                    fold_idx=fold_i,
                    random_state=global_random_state,
                    one_hot=True,
                )
        
            # Unpack the dataset with train test from a given fold
            X_train_pd, X_test_pd, y_train, y_test = dataset_preprocessor.get_data()
            
            # Convert to numpy
            X_train, X_test = [x.to_numpy() for x in (X_train_pd, X_test_pd)]
            # Random shuffle the data
            shuffle_indices = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]
            
            # Train M_1
            hparamsM1 = model_base_hyperparameters | model_fixed_hparams
            t0 = time.time()
            model1, pred_proba1, pred_crisp1 = train_model(X_train, y_train, model_type_to_use, model_fixed_seed, hparamsM1)
            time_model1 = time.time() - t0
            logging.info(f"Finished training M_1 in {time_model1} seconds")
                                            
            
            # Prepare Base Counterfactual Explainer
            base_explainer = prepare_base_counterfactual_explainer(
                base_cf_method=base_cf_method,
                hparams=CF_METHODS[base_cf_method],
                model=model1,
                X_train=X_train_pd,
                y_train=y_train,
                dataset_preprocessor=dataset_preprocessor,
                predict_fn_1_crisp=pred_crisp1
            )
            
            # Prepare the nearest neighbors model for the metrics
            nearest_neighbors_model = NearestNeighbors(n_neighbors=20, n_jobs=1)
            nearest_neighbors_model.fit(X_train)

            # Prepare 
            match robust_cf_method:
                case 'robx':
                    None 
                case 'uncertain_robx':
                    edlModel = train_EDL(X_train, y_train, global_random_state, ROBUST_CF_METHODS['uncertain_robx']['EDL'])
                case _:
                    None
        
            model2_handles = []
            model2_times = []
            
            if 'architecture' in perturbation_types.lower():
                hparams2_list = sample_architectures(m_count_per_experiment, model_hyperparameters_pool)
                # add base hyperparameters to the list
                hparams2_list = [model_base_hyperparameters | h for h in hparams2_list]
            else:
                hparams2_list = [model_base_hyperparameters | model_fixed_hparams] * m_count_per_experiment
                
            if 'bootstrap' in perturbation_types.lower():
                bootstrapM2 = sample_seeds(m_count_per_experiment)
            else:
                bootstrapM2 = [model_fixed_seed] * m_count_per_experiment
                
            if 'seed' in perturbation_types.lower():
                seedM2 = sample_seeds(m_count_per_experiment)
            else:
                seedM2 = [model_fixed_seed] * m_count_per_experiment
                
            
            for model_2_index in range(m_count_per_experiment):
                
                seed2 = seedM2[model_2_index]
                hparams2 = hparams2_list[model_2_index]
                bootstrap2seed = bootstrapM2[model_2_index]
                
                
                t0 = time.time() 
                model2, pred_proba2, pred_crisp2 = train_model_2(X_train, 
                    y_train, 
                    perturbation_types, 
                    model_type_to_use, 
                    hparams2,
                    seed2,
                    bootstrap2seed
                )
                time_model2 = time.time() - t0
                
                model2_handles.append((f"Model2_{model_2_index}", model2, pred_proba2, pred_crisp2))
                model2_times.append(time_model2)
                
            logging.info(f"Finished training {m_count_per_experiment} M_2 models")
            
            
            if EXPERIMENTS_SETUP['random_shuffle_test']:
                index_range = np.random.permutation(X_test.shape[0])
            else:
                index_range = range(X_test.shape[0])
            test_indices = index_range[:x_test_size]

            for x_index in test_indices:
                # Obtain the test sample
                x_test_sample = X_test[x_index]
                y_test_sample = y_test[x_index]
                x_test_sample_pd = pd.DataFrame(x_test_sample.reshape(1, -1), columns=X_test_pd.columns)
                
                # Obtain the predictions from M_1
                pred_proba1_sample = pred_proba1(x_test_sample)[0]
                pred_crisp1_sample = pred_crisp1(x_test_sample)[0]
                
                # If the prediction is 0, then the target class is 1, and vice versa
                taget_class = 1 - pred_crisp1_sample
                
                # Obtain the base counterfactual
                t0 = time.time()
                base_cf = base_counterfactual_generate(
                    base_explainer=base_explainer,
                    instance=x_test_sample_pd,
                )
                time_base_cf = time.time() - t0
                    
                # Calculate metrics
                base_metrics_model1 = calculate_metrics(
                    cf=base_cf,
                    cf_desired_class=taget_class,
                    x=x_test_sample,
                    y_train=y_train,
                    nearest_neighbors_model=nearest_neighbors_model,
                    predict_fn_crisp=pred_crisp1,
                )
                
                # This flag is just so that we insert the base counterfactual data only once in the results_df
                first_flag = True
                
                # If robx is used, then the beta_confidence and delta_robustness are not used so we set them to NaN
                match robust_cf_method:
                    case 'robx':
                        parameters = [(tau, vr) for tau in ROBUST_CF_METHODS['robx']['taus'] for vr in ROBUST_CF_METHODS['robx']['variances']]
                    case 'uncertain_robx':
                        parameters = [thresh for thresh in ROBUST_CF_METHODS['uncertain_robx']['thresh']]
                    case _:
                        parameters = [()]
                    
                # Loop over the beta_confidence and delta_robustness values and the M_2 models
                for params in parameters:
                    for model2_name, model2, pred_proba2, pred_crisp2 in model2_handles:
                        
                        # Start from calculating the validity of the base counterfactual  
                        # Do this only once as it is the same for all M_2 models and all beta_confidence and delta_robustness         
                        if first_flag:
                            if not check_is_none(base_cf):
                                base_cf_validity_model2 = int(int(pred_crisp2(base_cf)[0]) == taget_class)
                                base_counterfatual_model1_pred_proba = pred_proba1(base_cf)
                                base_counterfatual_model1_pred_crisp = pred_crisp1(base_cf)
                                base_counterfatual_model2_pred_proba = pred_proba2(base_cf)
                                base_counterfatual_model2_pred_crisp = pred_crisp2(base_cf)
                            else:
                                base_cf_validity_model2 = np.nan
                                base_counterfatual_model1_pred_proba = np.nan
                                base_counterfatual_model1_pred_crisp = np.nan
                                base_counterfatual_model2_pred_proba = np.nan
                                base_counterfatual_model2_pred_crisp = np.nan
                        base_cf_record = {
                            # Unique identifiers
                            'base_cf_method': base_cf_method,
                            'model_type_to_use': model_type_to_use,
                            'perturbation_types': perturbation_types,
                            'dataset_name': dataset_name,
                            'fold_i': fold_i,
                            'method_params': [params],
                            'model2_name': model2_name,
                            # Utility data
                            'x_test_sample': [x_test_sample],
                            'y_test_sample': [y_test_sample],
                            'model1_pred_proba': pred_proba1_sample,
                            'model1_pred_crisp': pred_crisp1_sample,
                            'model2_pred_proba': pred_proba2(x_test_sample),
                            'model2_pred_crisp': pred_crisp2(x_test_sample),
                            # Base counterfactual data
                            'base_counterfactual': [base_cf],
                            'base_counterfactual_model1_pred_proba': base_counterfatual_model1_pred_proba,
                            'base_counterfactual_model1_pred_crisp': base_counterfatual_model1_pred_crisp,
                            'base_counterfactual_model2_pred_proba': base_counterfatual_model2_pred_proba,
                            'base_counterfactual_model2_pred_crisp': base_counterfatual_model2_pred_crisp,
                            'base_counterfactual_validity': base_metrics_model1['validity'],
                            'base_counterfactual_proximityL1': base_metrics_model1['proximityL1'],
                            'base_counterfactual_proximityL2': base_metrics_model1['proximityL2'],
                            'base_counterfactual_plausibility': base_metrics_model1['plausibility'],
                            'base_counterfactual_discriminative_power': base_metrics_model1['dpow'],
                            'base_counterfactual_validity_model2': base_cf_validity_model2,
                            'base_counterfactual_time': time_base_cf,
                        }
                        
                        # Check if the base counterfactual is not None
                        if not check_is_none(base_cf):
                            t0 = time.time()
                            match robust_cf_method:
                                case 'robx':
                                    robust_counterfactual, _ = robx_algorithm(
                                        X_train = X_train,
                                        predict_class_proba_fn = pred_proba1,
                                        start_counterfactual = base_cf,
                                        tau = params[0], 
                                        variance = params[1],
                                        N = ROBUST_CF_METHODS['robx']['N'],
                                    )
                                    artifact_dict = {}
                                case 'uncertain_robx':
                                    robust_counterfactual = uncertain_robx_algorithm(
                                        X_train=X_train,
                                        predict_class_proba_fn=pred_proba1,
                                        start_counterfactual=base_cf,
                                        EDLmodel=edlModel,
                                        thresh=params,
                                        k=ROBUST_CF_METHODS['uncertain_robx']['k'],
                                    )
                                case _:
                                    raise ValueError('Unknown robust counterfactual method')
                            time_robust_cf = time.time() - t0
                        else:
                            robust_counterfactual = None
                            artifact_dict = {
                                'start_sample_passes_test': np.nan,
                                'counterfactual_does_not_pass_test': np.nan,
                                'counterfactual_does_not_have_target_class': np.nan,
                                'counterfactual_is_nan': np.nan,
                                'highest_delta': np.nan,
                            }
                            time_robust_cf = np.nan
                            
                        # Calculate the metrics
                        robust_metrics_model1 = calculate_metrics(
                            cf=robust_counterfactual,
                            cf_desired_class=taget_class,
                            x=x_test_sample,
                            y_train=y_train,
                            nearest_neighbors_model=nearest_neighbors_model,
                            predict_fn_crisp=pred_crisp1,
                        )
                        
                        if not check_is_none(robust_counterfactual):
                            robust_cf_validity_model2 = int(int(pred_crisp2(robust_counterfactual)[0]) == taget_class)
                            robust_cf_L1_distance_from_base_cf = np.sum(np.abs(robust_counterfactual - base_cf))
                            robust_cf_L2_distance_from_base_cf = np.sum(np.square(robust_counterfactual - base_cf))
                            robust_counterfactual_model1_pred_proba = pred_proba1(robust_counterfactual)
                            robust_counterfactual_model1_pred_crisp = pred_crisp1(robust_counterfactual)
                            robust_counterfactual_model2_pred_proba = pred_proba2(robust_counterfactual)
                            robust_counterfactual_model2_pred_crisp = pred_crisp2(robust_counterfactual)
                        else:
                            robust_cf_validity_model2 = np.nan
                            robust_cf_L1_distance_from_base_cf = np.nan
                            robust_cf_L2_distance_from_base_cf = np.nan
                            robust_counterfactual_model1_pred_proba = np.nan
                            robust_counterfactual_model1_pred_crisp = np.nan
                            robust_counterfactual_model2_pred_proba = np.nan
                            robust_counterfactual_model2_pred_crisp = np.nan
                        
                        # Store the results in the frame
                        robust_cf_record = {
                            # Robust counterfactual data
                            'robust_counterfactual': [robust_counterfactual],
                            'robust_counterfactual_model1_pred_proba': robust_counterfactual_model1_pred_proba,
                            'robust_counterfactual_model1_pred_crisp': robust_counterfactual_model1_pred_crisp,
                            'robust_counterfactual_model2_pred_proba': robust_counterfactual_model2_pred_proba,
                            'robust_counterfactual_model2_pred_crisp': robust_counterfactual_model2_pred_crisp,
                            'robust_counterfactual_validity': robust_metrics_model1['validity'],
                            'robust_counterfactual_proximityL1': robust_metrics_model1['proximityL1'],
                            'robust_counterfactual_proximityL2': robust_metrics_model1['proximityL2'],
                            'robust_counterfactual_plausibility': robust_metrics_model1['plausibility'],
                            'robust_counterfactual_discriminative_power': robust_metrics_model1['dpow'],
                            'robust_counterfactual_validity_model2': robust_cf_validity_model2,
                            'robust_counterfactual_L1_distance_from_base_cf': robust_cf_L1_distance_from_base_cf,
                            'robust_counterfactual_L2_distance_from_base_cf': robust_cf_L2_distance_from_base_cf,
                            'robust_counterfactual_time': time_robust_cf,
                        }
                        
                        # Add artifact_dict to the record
                        record = {**base_cf_record, **robust_cf_record, **artifact_dict}
                        record = pd.DataFrame(record, index=[0])
                        results_df = pd.concat([results_df, record], ignore_index=True)
            
                        # Save the results every n iterations            
                        if global_iteration % save_every_n_iterations == 0 and global_iteration > 0:
                            results_df.to_feather(f'./{results_df_dir}/{global_iteration}_results.feather')
                            cols = results_df.columns
                            
                            # Clear the results_df to save memory and speed up the process
                            del results_df
                            results_df = pd.DataFrame(columns=cols)
                            
                        global_iteration += 1
                        tqdm_pbar.update(1)
                        
                    first_flag = False

    # Final save                       
    results_df.to_feather(f'./{results_df_dir}/{global_iteration}_results.feather')
    # results_df = pd.DataFrame(columns=results_df.columns)

    # Progress bar close
    tqdm_pbar.close()
    
    # Save the miscellaneous data
    logging.info("Finished all experiments")
                
   
                


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the experiments')
    parser.add_argument('--config', type=str, default='./config.yml', help='The path to the config file')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    config = get_config(args.config)
    logging.debug(config)
    
    experiment(config)

