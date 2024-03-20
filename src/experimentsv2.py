import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import yaml
import os
import logging
import pandas as pd

from create_data_examples import Dataset, DatasetPreprocessor
from mlpclassifier import MLPClassifier, train_neural_network, train_K_mlps_in_parallel
from explainers import DiceExplainer, GrowingSpheresExplainer, BaseExplainer
from utils import bootstrap_data
from betarob import BetaRob

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
        seedB = None
    else:
        seedB = model_fixed_seed
        
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
        models = [model for partial_results in results for model in partial_results['models']]
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
          
def prepare_base_counterfactual_explainer(
        base_cf_method: str,
        hparams: dict,
        model: MLPClassifier,
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

def robust_counterfactual_generate(
        start_instance: np.ndarray | pd.DataFrame,
        target_class: int,
        delta_target: float,
        beta_confidence: float,
        dataset: Dataset, 
        preprocessor: DatasetPreprocessor, 
        pred_fn_crisp: callable,
        pred_fn_proba: callable,
        estimators_crisp: list[callable],
        grow_sphere_hparams: dict,
        classification_threshold: float,
        seed: int,
    ) -> tuple[np.ndarray, dict]:
    
    beta_explainer = BetaRob(
        dataset=dataset,
        preprocessor=preprocessor,
        pred_fn_crisp=pred_fn_crisp,
        pred_fn_proba=pred_fn_proba,
        estimators_crisp=estimators_crisp,
        grow_sphere_hparams=grow_sphere_hparams,
        classification_threshold=classification_threshold,
        seed=seed
    )
    
    beta_explainer.prep()
    
    robust_cf, artifact_dict = beta_explainer.generate(
        start_instance=start_instance,
        target_class=target_class,
        delta_target=delta_target,
        beta_confidence=beta_confidence
    )
    
    return robust_cf, artifact_dict
    
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

          
def experiment(config: dict, 
        robust_cf_method: str = 'betarob',
    ):
    
    GENERAL = config['general']
    EXPERIMENTS_SETUP = config['experiments_setup']
    MODEL_HYPERPARAMETERS = config['model_hyperparameters']
    BETA_ROB = config['beta_rob']
    
    # Extract the results directory
    results_dir = GENERAL['result_path']
    results_df_dir = os.path.join(results_dir, 'results')
    miscellaneous_df_path = os.path.join(results_dir, 'miscellaneous', 'miscellaneous_df.feather')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_df_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(miscellaneous_df_path)), exist_ok=True)
    
    
    # Extract the general parameters
    global_random_state = GENERAL['random_state']
    n_jobs = GENERAL['n_jobs']
    save_every_n_iterations = GENERAL['save_every_n_iterations']
    
    cv_folds = EXPERIMENTS_SETUP['cross_validation_folds']
    m_count_per_experiment = EXPERIMENTS_SETUP['m_count_per_experiment']
    x_test_size = EXPERIMENTS_SETUP['x_test_size']
    ex_types = EXPERIMENTS_SETUP['ex_types']
    datasets = EXPERIMENTS_SETUP['datasets']
    beta_confidences = EXPERIMENTS_SETUP['beta_confidences']
    delta_robustnesses = EXPERIMENTS_SETUP['delta_robustnesses'] 
    model_type_to_use = EXPERIMENTS_SETUP['model_type_to_use']
    base_cf_method = EXPERIMENTS_SETUP['base_counterfactual_method']
    
    # Extract the beta-robustness parameters
    k_mlps_in_B = BETA_ROB['k_mlps_in_B']
    beta_gs_hparams = BETA_ROB['growingSpheresHparams']
    
    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_seed']
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_hyperparameters']
    model_hyperparameters_pool = MODEL_HYPERPARAMETERS[model_type_to_use]['model_hyperparameters_pool']
    model_base_hyperparameters = MODEL_HYPERPARAMETERS[model_type_to_use]['model_base_hyperparameters']
    
    classification_threshold = model_base_hyperparameters['classification_threshold']
    
    
  
    results_df = pd.DataFrame()
    miscellaneous_df = pd.DataFrame()
    
    global_iteration = 0
    all_iterations = len(ex_types) * len(datasets) * cv_folds * m_count_per_experiment * len(ex_types) * x_test_size * len(beta_confidences) * len(delta_robustnesses)
    tqdm_pbar = tqdm(total=all_iterations, desc="Overall progress")
    
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
                    one_hot=True,
                )
            
                # Unpack the dataset with train test from a given fold
                X_train_pd, X_test_pd, y_train, y_test = dataset_preprocessor.get_data()
                
                # Convert to numpy
                X_train, X_test = [x.to_numpy() for x in (X_train_pd, X_test_pd)]
                
                # Train M_1
                hparamsM1 = model_base_hyperparameters | model_fixed_hparams
                t0 = time.time()
                model1, pred_proba1, pred_crisp1 = train_model(X_train, y_train, model_type_to_use, model_fixed_seed, hparamsM1)
                time_model1 = time.time() - t0
                logging.info(f"Finished training M_1 in {time_model1} seconds")
                
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
                modelsB_crisp_fns = [lambda x: model.predict_crisp(x, classification_threshold) for model in modelsB]
                logging.info(f"Finished training B in {time_modelsB} seconds")
                                                
                # Add miscellaneous data to the frame
                record = {
                    'experiment_type': ex_type,
                    'dataset_name': dataset_name,
                    'fold_i': fold_i,
                    'model1_time': time_model1,
                    'modelsB_time': time_modelsB,
                }
                record = pd.DataFrame(record, index=[0])
                miscellaneous_df = pd.concat([miscellaneous_df, record], ignore_index=True)
                # Save the miscellaneous data
                miscellaneous_df.to_feather(miscellaneous_df_path)
                
                # Prepare Base Counterfactual Explainer
                base_explainer = prepare_base_counterfactual_explainer(
                    base_cf_method=base_cf_method,
                    hparams=beta_gs_hparams,
                    model=model1,
                    X_train=X_train_pd,
                    y_train=y_train,
                    dataset_preprocessor=dataset_preprocessor,
                    predict_fn_1_crisp=pred_crisp1
                )
                
                # Prepare the nearest neighbors model for the metrics
                nearest_neighbors_model = NearestNeighbors(n_neighbors=20, n_jobs=1)
                nearest_neighbors_model.fit(X_train)
                    
                
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
                        
                        if 'seed' in ex_type.lower(): # append the fixed seed also if needed
                            hparams2['model_fixed_seed'] = model_fixed_seed
                        
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
                        
                        for beta_confidence in beta_confidences:
                            for delta_robustness in delta_robustnesses:
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
                                        record = {
                                            # Unique identifiers
                                            'experiment_type': ex_type,
                                            'dataset_name': dataset_name,
                                            'fold_i': fold_i,
                                            'experiment_generalization_type': ex_generalization,
                                            'beta_confidence': beta_confidence,
                                            'delta_robustness': delta_robustness,
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
                                        record = pd.DataFrame(record, index=[0])
                                        results_df = pd.concat([results_df, record], ignore_index=True)
                                    
                                    # Obtain the predictions from M_2
                                    pred_proba2_sample = pred_proba2(x_test_sample)[0]
                                    pred_crisp2_sample = pred_crisp2(x_test_sample)[0]
                                    
                                    # Obtain the robust counterfactual
                                    if not check_is_none(base_cf):
                                        t0 = time.time()
                                        match robust_cf_method:
                                            case 'betarob':
                                                robust_counterfactual, artifact_dict = robust_counterfactual_generate(
                                                    start_instance=base_cf,
                                                    target_class=taget_class,
                                                    delta_target=delta_robustness,
                                                    beta_confidence=beta_confidence,
                                                    dataset=dataset, 
                                                    preprocessor=dataset_preprocessor, 
                                                    pred_fn_crisp=pred_crisp1,
                                                    pred_fn_proba=pred_proba1,
                                                    estimators_crisp=modelsB_crisp_fns,
                                                    grow_sphere_hparams=beta_gs_hparams,
                                                    classification_threshold=classification_threshold,
                                                    seed=global_random_state
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
                                    record = {
                                        # Unique identifiers
                                        'experiment_type': ex_type,
                                        'dataset_name': dataset_name,
                                        'fold_i': fold_i,
                                        'experiment_generalization_type': ex_generalization,
                                        'beta_confidence': beta_confidence,
                                        'delta_robustness': delta_robustness,
                                        'model2_name': model2_name,
                                        # Utility data
                                        'x_test_sample': [x_test_sample],
                                        'y_test_sample': [y_test_sample],
                                        'model1_pred_proba': pred_proba1_sample,
                                        'model1_pred_crisp': pred_crisp1_sample,
                                        'model2_pred_proba': pred_proba2_sample,
                                        'model2_pred_crisp': pred_crisp2_sample,
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
                                    record = {**record, **artifact_dict}
                                    record = pd.DataFrame(record, index=[0])
                                    results_df = pd.concat([results_df, record], ignore_index=True)
                        
                                    # Save the results every n iterations            
                                    if global_iteration % save_every_n_iterations == 0:
                                        results_df.to_feather(f'./{results_dir}/{global_iteration}_results.feather')
                                        
                                        # Clear the results_df to save memory and speed up the process
                                        results_df = pd.DataFrame(columns=results_df.columns)
                                        
                                    global_iteration += 1
                                    tqdm_pbar.update(1)
                                    
                                first_flag = False
    
    # Final save                       
    results_df.to_feather(f'./{results_dir}/{global_iteration}_results.feather')
    results_df = pd.DataFrame(columns=results_df.columns)
    
    # Progress bar close
    tqdm_pbar.close()
    
    # Save the miscellaneous data
    logging.info("Finished all experiments")
                
   
                


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)
    config = get_config()
    logging.debug(config)
    
    experiment(config)
