import os
import time
import yaml
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors

# Project imports
from .experiments_utils import *
          
def experiment(config: dict):
    
    GENERAL = config['general']
    EXPERIMENTS_SETUP = config['experiments_setup']
    MODEL_HYPERPARAMETERS = config['model_hyperparameters']
    BETA_ROB = config['beta_rob']
    
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
    cv_folds = EXPERIMENTS_SETUP['cross_validation_folds']
    m_count_per_experiment = EXPERIMENTS_SETUP['m_count_per_experiment']
    x_test_size = EXPERIMENTS_SETUP['x_test_size']
    ex_types = EXPERIMENTS_SETUP['ex_types']
    datasets = EXPERIMENTS_SETUP['datasets']
    model_type_to_use = EXPERIMENTS_SETUP['model_type_to_use']
    base_cf_method = EXPERIMENTS_SETUP['base_counterfactual_method']
    perform_generalizations = EXPERIMENTS_SETUP['perform_generalizations']
    just_base_cf = EXPERIMENTS_SETUP['just_base_cf']
    
    # Extract the beta-robustness parameters
    k_mlps_in_B_options = BETA_ROB['k_mlps_in_B']
    beta_gs_hparams = BETA_ROB['growingSpheresHparams']
    
    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_seed']
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use]['model_fixed_hyperparameters']
    model_hyperparameters_pool = MODEL_HYPERPARAMETERS[model_type_to_use]['model_hyperparameters_pool']
    model_base_hyperparameters = MODEL_HYPERPARAMETERS[model_type_to_use]['model_base_hyperparameters']
   
    if robust_cf_method == 'betarob':
        hparams = beta_gs_hparams
    elif robust_cf_method == 'robx':
        hparams = {}
    else:
        raise ValueError('Unknown robust counterfactual method')
        
    if base_cf_method == 'roar':
        hparams = config['roar']
        all_combinations = []
        for dmx in hparams['delta_max']:
            for lr in hparams['lr']:            
                for norm in hparams['norm']:
                    _base = deepcopy(hparams)
                    _base['delta_max'] = dmx
                    _base['lr'] = lr
                    _base['norm'] = norm
                    hparams_to_save = _base.copy()
                    all_combinations.append((_base, hparams_to_save))
    elif base_cf_method == 'rbr':
        hparams: dict = config['rbr']
        all_combinations = []
        for pr in hparams['perturb_radius']['synthesis']:
            for dp in hparams['ec']['rbr_params']['delta_plus']:
                for s in hparams['ec']['rbr_params']['sigma']:
                    _base = deepcopy(hparams)
                    _base['perturb_radius']['synthesis'] = pr
                    _base['ec']['rbr_params']['delta_plus'] = dp
                    _base['ec']['rbr_params']['sigma'] = s
                    hparams_to_save = {
                        'perturb_radius': pr,
                        'delta_plus': dp,
                        'sigma': s
                    }
                    all_combinations.append((_base, hparams_to_save))
    elif base_cf_method == 'face':
        hparams = config['face']
        all_combinations = []
        for f in hparams['fraction']:
            for m in hparams['mode']:
                _base = deepcopy(hparams)
                _base['fraction'] = f
                _base['mode'] = m
                hp_to_save = _base.copy()
                all_combinations.append((_base, hp_to_save))
    elif base_cf_method == 'gs':
        hparams = config['gs']
        all_combinations = [(hparams, hparams)]
    elif base_cf_method == 'dice':
        hparams = config['dice']
        all_combinations = []
        for pw in hparams['proximity_weight']:
            for dw in hparams['diversity_weight']:
                for ct in hparams['sparsity_weight']:
                    _base = deepcopy(hparams)
                    _base['proximity_weight'] = pw
                    _base['diversity_weight'] = dw
                    _base['sparsity_weight'] = ct
                    hparams_to_save = _base.copy()
                    all_combinations.append((_base, hparams_to_save))
    else:
        raise ValueError('Unknown base counterfactual method')
   
    results_df = pd.DataFrame()
    
    global_iteration = 0
    all_iterations = len(ex_types) * len(datasets) * cv_folds * m_count_per_experiment  * x_test_size * len(k_mlps_in_B_options) * len(all_combinations)
    
        
    tqdm_pbar = tqdm(total=all_iterations, desc="Overall progress")
    
    for ex_type in ex_types:
        logging.info(f"Running experiment type: {ex_type}")
        
        for dataset_name in datasets:
            logging.info(f"Running experiment for dataset: {dataset_name}")
            
            for k_mlps_in_B in k_mlps_in_B_options:
            
                dataset = Dataset(dataset_name)
                for (hparams, hparams_to_save) in all_combinations:
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
                                                        
                        
                        # Prepare Base Counterfactual Explainer
                        base_explainer = prepare_base_counterfactual_explainer(
                            base_cf_method=base_cf_method,
                            hparams=hparams,
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
                                try:
                                    base_cf = base_counterfactual_generate(
                                        base_explainer=base_explainer,
                                        instance=x_test_sample_pd,
                                    )
                                    if check_is_none(base_cf):
                                        base_cf = None
                                except:
                                    base_cf = None
                                    logging.warning('BASE CF NOT FOUND')
                                    
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
                                
                                if not just_base_cf:
                                    raise ValueError('THIS SCRIPT IS JUST FOR BASE CF')
                                
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
                                        'just_base_cf': just_base_cf,
                                        'k_mlps_in_B': k_mlps_in_B,
                                        'fold_i': fold_i,
                                        'experiment_generalization_type': ex_generalization,
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
                                    
                                    

                                    robust_counterfactual = None
                                    artifact_dict = {
                                        'start_sample_passes_test': np.nan,
                                        'counterfactual_does_not_pass_test': np.nan,
                                        'counterfactual_does_not_have_target_class': np.nan,
                                        'counterfactual_is_nan': np.nan,
                                        'highest_delta': np.nan,
                                    }
                                    time_robust_cf = np.nan
                                    
                                    # Store the results in the frame
                                    robust_cf_record = {
                                        # Robust counterfactual data
                                        'robust_counterfactual': [robust_counterfactual],
                                        'robust_counterfactual_model1_pred_proba': np.nan,
                                        'robust_counterfactual_model1_pred_crisp': np.nan,
                                        'robust_counterfactual_model2_pred_proba': np.nan,
                                        'robust_counterfactual_model2_pred_crisp': np.nan,
                                        'robust_counterfactual_validity': np.nan,
                                        'robust_counterfactual_proximityL1': np.nan,
                                        'robust_counterfactual_proximityL2': np.nan,
                                        'robust_counterfactual_plausibility': np.nan,
                                        'robust_counterfactual_discriminative_power': np.nan,
                                        'robust_counterfactual_validity_model2': np.nan,
                                        'robust_counterfactual_L1_distance_from_base_cf': np.nan,
                                        'robust_counterfactual_L2_distance_from_base_cf': np.nan,
                                        'robust_counterfactual_time': time_robust_cf,
                                    }
                                    # Add artifact_dict to the record
                                    record = {**base_cf_record, **hparams_to_save, **robust_cf_record, **artifact_dict}
                                    record = pd.DataFrame(record, index=[0])
                                    results_df = pd.concat([results_df, record], ignore_index=True)
                        
                                    # Save the results every n iterations            
                                    if global_iteration % save_every_n_iterations == 0 and global_iteration > 0:
                                        # results_df.to_feather(f'./{results_df_dir}/{global_iteration}_results.feather')
                                        results_df.to_csv(f'{results_df_dir}/{global_iteration}_results.csv')
                                        cols = results_df.columns
                                        
                                        # Clear the results_df to save memory and speed up the process
                                        del results_df
                                        results_df = pd.DataFrame(columns=cols)
                                        
                                    global_iteration += 1
                                    tqdm_pbar.update(1)
                                    
                                first_flag = False
            
    # Final save                       
    # results_df.to_feather(f'./{results_df_dir}/{global_iteration}_results.feather')
    results_df.to_csv(f'{results_df_dir}/{global_iteration}_results.csv')
    # results_df = pd.DataFrame(columns=results_df.columns)
    
    # Progress bar close
    tqdm_pbar.close()
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

