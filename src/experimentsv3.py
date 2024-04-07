import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy import stats
import yaml
import os
import logging
import pandas as pd

from create_data_examples import Dataset, DatasetPreprocessor
from mlpclassifier import MLPClassifier, train_neural_network, train_K_mlps_in_parallel
from rfclassifier import RFClassifier, train_random_forest, train_K_rfs_in_parallel
from dtclassifier import DecisionTree, train_decision_tree, train_K_dts_in_parallel
from baseclassifier import BaseClassifier
from explainers import DiceExplainer, GrowingSpheresExplainer, BaseExplainer
from robx import robx_algorithm
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
        m, pp, pc = train_random_forest(X_train, y_train, seed, hparams)
    elif model_type == 'decision_tree':
        m, pp, pc = train_decision_tree(X_train, y_train, seed, hparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return m, pp, pc
        
def train_B(ex_type: str,
        model_type_to_use: str,
        model_base_hyperparameters: dict,
        seedB: list[int],
        hparamsB: list[dict],   
        bootstrapB: list[int],
        k_mlps_in_B: int,
        n_jobs: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> list:
    
    
    match model_type_to_use:
        case 'neural_network':
            results = train_K_mlps_in_parallel(X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hparamsB=hparamsB,
                bootstrapB=bootstrapB,
                seedB=seedB,
                hparams_base=model_base_hyperparameters,
                ex_type=ex_type,
                K=k_mlps_in_B,
                n_jobs=n_jobs,
            )
            models = [model for partial_results in results for model in partial_results['models']]
        # case 'random_forest':
        #     results = train_K_rfs_in_parallel(X_train=X_train,
        #         y_train=y_train,
        #         X_test=X_test,
        #         y_test=y_test,
        #         hparams=hparamsB,
        #         bootstrap=bootstrapB,
        #         fixed_hparams=fixed_hparams,
        #         fixed_seed=seedB,
        #         K=k_mlps_in_B,
        #         n_jobs=n_jobs,
        #     )
        #     models = [model for partial_results in results for model in partial_results['models']]   
        # case 'decision_tree':
        #     results = train_K_dts_in_parallel(X_train=X_train,
        #         y_train=y_train,
        #         X_test=X_test,
        #         y_test=y_test,
        #         hparams=hparamsB,
        #         bootstrap=bootstrapB,
        #         fixed_hparams=fixed_hparams,
        #         fixed_seed=seedB,
        #         K=k_mlps_in_B,
        #         n_jobs=n_jobs,
        #     )
        #     models = [model for partial_results in results for model in partial_results['models']]
        case _:
            raise ValueError('Unknown model type. Cannot train B models.')
        
    return models
        
def train_model_2(X_train, y_train,
        ex_type: str,
        model_type: str,
        hparams: dict,
        seed: int,
        bootstrap_seed: int,
    ) -> tuple:
    
    # Perform bootrap with the bootstrap seed
    if 'bootstrap' in ex_type.lower():
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

def is_robustness_achievable_for_params(k: int, beta_confidence: float, delta_robustness: float) -> bool:
    '''
    Check if with the given parameters the robustness is achievable.
    '''
    lb, _ = stats.beta.interval(beta_confidence, 0.5 + k, 0.5)
    return lb > delta_robustness
    
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
                architecture[_param] = np.random.randint(lower, upper + 1)
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


          
def experiment(config: dict):
    
    GENERAL = config['general']
    EXPERIMENTS_SETUP = config['experiments_setup']
    MODEL_HYPERPARAMETERS = config['model_hyperparameters']
    BETA_ROB = config['beta_rob']
    ROBX = config['robx']
    
    # Extract the results directory
    results_dir = GENERAL['result_path']
    results_df_dir = os.path.join(results_dir, 'results')
    miscellaneous_df_dir = os.path.join(results_dir, 'miscellaneous')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_df_dir, exist_ok=True)
    os.makedirs(miscellaneous_df_dir, exist_ok=True)
    
    # Save the config right away
    with open(os.path.join(results_dir, 'config.yml'), 'w') as file:
        yaml.dump(config, file)
    
    
    # Extract the general parameters
    global_random_state = GENERAL['random_state']
    n_jobs = GENERAL['n_jobs']
    save_every_n_iterations = GENERAL['save_every_n_iterations']
    
    robust_cf_method = EXPERIMENTS_SETUP['robust_cf_method']
    cv_folds = EXPERIMENTS_SETUP['cross_validation_folds']
    m_count_per_experiment = EXPERIMENTS_SETUP['m_count_per_experiment']
    x_test_size = EXPERIMENTS_SETUP['x_test_size']
    ex_types = EXPERIMENTS_SETUP['ex_types']
    datasets = EXPERIMENTS_SETUP['datasets']
    beta_confidences = EXPERIMENTS_SETUP['beta_confidences']
    delta_robustnesses = EXPERIMENTS_SETUP['delta_robustnesses'] 
    model_type_to_use = EXPERIMENTS_SETUP['model_type_to_use']
    base_cf_method = EXPERIMENTS_SETUP['base_counterfactual_method']
    perform_generalizations = EXPERIMENTS_SETUP['perform_generalizations']
    
    # Extract the beta-robustness parameters
    k_mlps_in_B_options = BETA_ROB['k_mlps_in_B']
    beta_gs_hparams = BETA_ROB['growingSpheresHparams']
    
    # Extract the robX parameters
    robx_taus = ROBX['taus']
    robx_variances = ROBX['variances']
    robx_N = ROBX['N']
    
    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_seed']
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_hyperparameters']
    model_hyperparameters_pool = MODEL_HYPERPARAMETERS[model_type_to_use]['model_hyperparameters_pool']
    model_base_hyperparameters = MODEL_HYPERPARAMETERS[model_type_to_use]['model_base_hyperparameters']
    
    classification_threshold = model_base_hyperparameters['classification_threshold']
   
    results_df = pd.DataFrame()
    miscellaneous_df = pd.DataFrame()
    
    global_iteration = 0
    all_iterations = len(ex_types) * len(datasets) * cv_folds * m_count_per_experiment  * x_test_size * len(k_mlps_in_B_options)
    
    if perform_generalizations: # If generalizations are performed, then multiply the iterations by the number of generalizations
        all_iterations *= len(ex_types)
    
    if robust_cf_method == 'robx': # If robx is used, then the iterations are multiplied by the number of taus
        all_iterations *= len(robx_taus) * len(robx_variances)
    
    if robust_cf_method == 'betarob': # If beta-robustness is used, then the iterations are multiplied by the number of beta_confidences and delta_robustnesses
        all_iterations *= len(beta_confidences) * len(delta_robustnesses)
        
    tqdm_pbar = tqdm(total=all_iterations, desc="Overall progress")
    
    for ex_type in ex_types:
        logging.info(f"Running experiment type: {ex_type}")
        
        for dataset_name in datasets:
            logging.info(f"Running experiment for dataset: {dataset_name}")
            
            for k_mlps_in_B in k_mlps_in_B_options:
            
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
                    # Random shuffle the data
                    shuffle_indices = np.random.permutation(X_train.shape[0])
                    X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]
                    
                    # Train M_1
                    hparamsM1 = model_base_hyperparameters | model_fixed_hparams
                    t0 = time.time()
                    model1, pred_proba1, pred_crisp1 = train_model(X_train, y_train, model_type_to_use, model_fixed_seed, hparamsM1)
                    time_model1 = time.time() - t0
                    logging.info(f"Finished training M_1 in {time_model1} seconds")
                    
                    
                    # Should bootstrap vary?
                    if 'bootstrap' in ex_type.lower():
                        bootstrapB = sample_seeds(k_mlps_in_B)
                    else:
                        bootstrapB = [model_fixed_seed] * k_mlps_in_B
                        
                    # Should seed vary?
                    if 'seed' in ex_type.lower():
                        seedB = sample_seeds(k_mlps_in_B)
                    else:
                        seedB = [model_fixed_seed] * k_mlps_in_B
                        
                    # Should architecture vary? 
                    if 'architecture' in ex_type.lower():
                        hparamsB = sample_architectures(k_mlps_in_B, model_hyperparameters_pool)
                    else:
                        hparamsB = [model_fixed_hparams] * k_mlps_in_B
                    
                    # Train B
                    t0 = time.time()
                    modelsB = train_B(
                        ex_type=ex_type,
                        model_type_to_use=model_type_to_use,
                        model_base_hyperparameters=model_base_hyperparameters,
                        seedB=seedB,
                        hparamsB=hparamsB,
                        bootstrapB=bootstrapB,
                        k_mlps_in_B=k_mlps_in_B,
                        n_jobs=n_jobs,
                        X_train=X_train, 
                        y_train=y_train, 
                        X_test=X_test, 
                        y_test=y_test
                    )
                    time_modelsB = time.time() - t0
                    # modelsB_crisp_fns = [lambda x: model.predict_crisp(x, classification_threshold) for model in modelsB]
                    modelsB_crisp_fns = [model.predict_crisp for model in modelsB]
                    logging.info(f"Finished training B in {time_modelsB} seconds")
                                                    
                    # Add miscellaneous data to the frame
                    record = {
                        'experiment_type': ex_type,
                        'dataset_name': dataset_name,
                        'k_mlps_in_B': k_mlps_in_B,
                        'fold_i': fold_i,
                        'model1_time': time_model1,
                        'modelsB_time': time_modelsB,
                    }
                    record = pd.DataFrame(record, index=[0])
                    miscellaneous_df = pd.concat([miscellaneous_df, record], ignore_index=True)
                    
                    # Save the miscellaneous data
                    if len(miscellaneous_df) >= save_every_n_iterations:
                        path = os.path.join(miscellaneous_df_dir, f'miscellaneous_df_{global_iteration}.feather')
                        miscellaneous_df.to_feather(path)
                        
                        # Preserve memory
                        cols = miscellaneous_df.columns
                        del miscellaneous_df
                        miscellaneous_df = pd.DataFrame(cols=cols)
                    
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
                    
                    # If generalizations are not performed, then use the ex_type as the only generalization type 
                    # -- which is not a generalization as it is the same as the experiment type
                    if not perform_generalizations:
                        ex_types_for_generatilaztion = [ex_type]
                    else:
                        ex_types_for_generatilaztion = ex_types
       
                    # Run the experiments for the generalization types
                    for ex_generalization in ex_types_for_generatilaztion:
                        logging.info(f"Running experiment for generalization type: {ex_generalization}")
                        
                        model2_handles = []
                        model2_times = []
                        
                        if 'architecture' in ex_generalization.lower():
                            hparams2_list = sample_architectures(m_count_per_experiment, model_hyperparameters_pool)
                            # add base hyperparameters to the list
                            hparams2_list = [model_base_hyperparameters | h for h in hparams2_list]
                        else:
                            hparams2_list = [model_base_hyperparameters | model_fixed_hparams] * m_count_per_experiment
                            
                        if 'bootstrap' in ex_generalization.lower():
                            bootstrapM2 = sample_seeds(m_count_per_experiment)
                        else:
                            bootstrapM2 = [model_fixed_seed] * m_count_per_experiment
                            
                        if 'seed' in ex_generalization.lower():
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
                                ex_generalization, 
                                model_type_to_use, 
                                hparams2,
                                seed2,
                                bootstrap2seed
                            )
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
                            
                            # If robx is used, then the beta_confidence and delta_robustness are not used so we set them to NaN
                            if robust_cf_method == 'robx':
                                _beta_confidences = robx_taus # use the taus for robx, but to simplify the code, we just assign it to beta_confidences variable
                                _delta_robustnesses =  robx_variances # use the variances for robx, but to simplify the code, we just assign it to delta_robustnesses variable
                            else: # otherwise, use the given values in config
                                _beta_confidences = beta_confidences
                                _delta_robustnesses = delta_robustnesses
                            
                            # Loop over the beta_confidence and delta_robustness values and the M_2 models
                            for beta_confidence in _beta_confidences:
                                for delta_robustness in _delta_robustnesses:
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
                                            'experiment_type': ex_type,
                                            'dataset_name': dataset_name,
                                            'k_mlps_in_B': k_mlps_in_B,
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
                                        
                                        # Obtain the robust counterfactual
                                        # Check if the robustness is achievable and if the base counterfactual is not None
                                        achievable = is_robustness_achievable_for_params(k_mlps_in_B, beta_confidence, delta_robustness)
                                        isNone = check_is_none(base_cf)
                                        if (achievable or not achievable and robust_cf_method == 'robx') and not isNone:
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
                                                case 'robx':
                                                    robust_counterfactual, _ = robx_algorithm(
                                                        X_train = X_train,
                                                        predict_class_proba_fn = pred_proba1,
                                                        start_counterfactual = base_cf,
                                                        tau = beta_confidence, # This is the tau parameter in the robx algorithm, just reusing the beta_confidence for simplicity
                                                        variance = delta_robustness,
                                                        N = robx_N,
                                                    )
                                                    artifact_dict = {}
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
    
    # Final miscellaneous save
    miscellaneous_df.to_feather(f'./{miscellaneous_df_dir}/{global_iteration}_miscellaneous.feather')
    # miscellaneous_df = pd.DataFrame(columns=miscellaneous_df.columns)
    
    # Progress bar close
    tqdm_pbar.close()
    
    # Save the miscellaneous data
    logging.info("Finished all experiments")
                
   
                


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the experiments')
    parser.add_argument('--config', type=str, default='./configv2.yml', help='The path to the config file')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    config = get_config(args.config)
    logging.debug(config)
    
    experiment(config)

