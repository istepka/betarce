import os
import time
import yaml
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Project imports
from .explainers.posthoc import robx_algorithm
from .datasets import Dataset, DatasetPreprocessor
from .experiments_utils import (
    train_model,
    train_B,
    train_model_2,
    prepare_base_counterfactual_explainer,
    base_counterfactual_generate,
    betarce_generate,
    calculate_metrics,
    is_robustness_achievable_for_params,
    check_is_none,
    sample_seeds,
    sample_architectures,
    get_B,
)

# log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


def experiment(config: dict):
    GENERAL = config["general"]
    EXPERIMENTS_SETUP = config["experiments_setup"]
    MODEL_HYPERPARAMETERS = config["model_hyperparameters"]
    BETA_ROB = config["beta_rob"]
    ROBX = config["robx"]

    mm_dd = time.strftime("%m-%d")
    salted_hash = str(abs(hash(str(config))) + int(time.time()))[:5]
    prefix = f"{mm_dd}_{salted_hash}"
    GENERAL["result_path"] = os.path.join(GENERAL["result_path"], prefix)
    GENERAL["model_path"] = os.path.join(GENERAL["model_path"], prefix)
    GENERAL["log_path"] = os.path.join(GENERAL["log_path"], prefix)

    logging.debug(config)

    # Extract the results directory
    results_dir = GENERAL["result_path"]
    results_df_dir = os.path.join(results_dir, "results")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_df_dir, exist_ok=True)

    # Save the config right away
    with open(os.path.join(results_dir, "config.yml"), "w") as file:
        yaml.dump(config, file)

    # Extract the general parameters
    global_random_state = GENERAL["random_state"]
    n_jobs = GENERAL["n_jobs"]
    save_every_n_iterations = GENERAL["save_every_n_iterations"]

    robust_cf_method = EXPERIMENTS_SETUP["robust_cf_method"]
    cv_folds = EXPERIMENTS_SETUP["cross_validation_folds"]
    m_count_per_experiment = EXPERIMENTS_SETUP["m_count_per_experiment"]
    x_test_size = EXPERIMENTS_SETUP["x_test_size"]
    ex_types = EXPERIMENTS_SETUP["ex_types"]
    datasets = EXPERIMENTS_SETUP["datasets"]
    beta_confidences = EXPERIMENTS_SETUP["beta_confidences"]
    delta_robustnesses = EXPERIMENTS_SETUP["delta_robustnesses"]
    model_type_to_use = EXPERIMENTS_SETUP["model_type_to_use"]
    base_cf_method = EXPERIMENTS_SETUP["base_counterfactual_method"]
    perform_generalizations = EXPERIMENTS_SETUP["perform_generalizations"]
    just_base_cf = EXPERIMENTS_SETUP["just_base_cf"]

    # Extract the beta-robustness parameters
    k_mlps_in_B_options = BETA_ROB["k_mlps_in_B"]
    beta_gs_hparams = BETA_ROB["growingSpheresHparams"]

    # Extract the robX parameters
    robx_taus = ROBX["taus"]
    robx_variances = ROBX["variances"]
    robx_N = ROBX["N"]

    # Get the model hyperparameters
    model_fixed_seed = MODEL_HYPERPARAMETERS[model_type_to_use]["model_fixed_seed"]
    model_fixed_hparams = MODEL_HYPERPARAMETERS[model_type_to_use][
        "model_fixed_hyperparameters"
    ]
    model_hp_pool = MODEL_HYPERPARAMETERS[model_type_to_use][
        "model_hyperparameters_pool"
    ]
    model_base_hp = MODEL_HYPERPARAMETERS[model_type_to_use][
        "model_base_hyperparameters"
    ]

    classification_threshold = model_base_hp["classification_threshold"]

    results_df = pd.DataFrame()

    global_iteration = 0
    all_iterations = (
        len(ex_types)
        * len(datasets)
        * cv_folds
        * m_count_per_experiment
        * x_test_size
        * len(k_mlps_in_B_options)
    )
    # Log each number
    logging.info(f"Number of experiments: {len(ex_types)}")
    logging.info(f"Number of datasets: {len(datasets)}")
    logging.info(f"Number of cross-validation folds: {cv_folds}")
    logging.info(f"Number of M_2 models per experiment: {m_count_per_experiment}")
    logging.info(f"Number of test samples: {x_test_size}")
    logging.info(f"Number of k_mlps_in_B options: {len(k_mlps_in_B_options)}")

    if perform_generalizations:  # If generalizations are performed, then multiply the iterations by the number of generalizations
        all_iterations *= 3  # TODO: Make it dynamic
        logging.info("Number of generalizations: 3")

    logging.info(f"Total number of iterations: {all_iterations}")

    if not just_base_cf:
        if (
            robust_cf_method == "robx"
        ):  # If robx is used, then the iterations are multiplied by the number of taus
            all_iterations *= len(robx_taus) * len(robx_variances)

        if (
            robust_cf_method == "betarob"
        ):  # If beta-robustness is used, then the iterations are multiplied by the number of beta_confidences and delta_robustnesses
            all_iterations *= len(beta_confidences) * len(delta_robustnesses)

    tqdm_pbar = tqdm(total=all_iterations, desc="Overall progress")

    to_make_combinations = [ex_types, datasets, k_mlps_in_B_options, range(cv_folds)]
    all_combinations = np.array(np.meshgrid(*to_make_combinations)).T.reshape(
        -1, len(to_make_combinations)
    )

    for combination in all_combinations:
        ex_type, dataset_name, k_mlps_in_B, fold_i = combination
        fold_i = int(fold_i)
        k_mlps_in_B = int(k_mlps_in_B)

        logging.info(f"Running experiment type: {ex_type}")

        logging.info(f"Running experiment for dataset: {dataset_name}")

        dataset = Dataset(dataset_name)

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
        X_train, y_train = (
            X_train[shuffle_indices],
            y_train[shuffle_indices],
        )

        # Train M_1
        hparamsM1 = model_base_hp | model_fixed_hparams
        t0 = time.time()
        model1, pred_proba1, pred_crisp1 = train_model(
            X_train, y_train, model_type_to_use, model_fixed_seed, hparamsM1
        )
        time_model1 = time.time() - t0
        logging.info(f"Finished training M_1 in {time_model1} seconds")

        hparamsB, seedB, bootstrapB = get_B(
            ex_type,
            k_mlps_in_B,
            model_fixed_hparams,
            model_hp_pool,
            model_fixed_seed,
        )

        # Train B
        t0 = time.time()
        modelsB = train_B(
            ex_type=ex_type,
            model_type_to_use=model_type_to_use,
            model_base_hyperparameters=model_base_hp,
            seedB=seedB,
            hparamsB=hparamsB,
            bootstrapB=bootstrapB,
            k_mlps_in_B=k_mlps_in_B,
            n_jobs=n_jobs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        time_modelsB = time.time() - t0
        # modelsB_crisp_fns = [lambda x: model.predict_crisp(x, classification_threshold) for model in modelsB]
        modelsB_crisp_fns = [model.predict_crisp for model in modelsB]
        logging.info(f"Finished training B in {time_modelsB} seconds")

        # Prepare Base Counterfactual Explainer
        base_explainer = prepare_base_counterfactual_explainer(
            base_cf_method=base_cf_method,
            hparams=beta_gs_hparams,
            model=model1,
            X_train=X_train_pd,
            y_train=y_train,
            dataset_preprocessor=dataset_preprocessor,
            predict_fn_1_crisp=pred_crisp1,
        )

        # Prepare the nearest neighbors model for the metrics
        nearest_neighbors_model = NearestNeighbors(n_neighbors=20, n_jobs=1)
        nearest_neighbors_model.fit(X_train)

        # If generalizations are not performed, then use the ex_type as the only generalization type
        # -- which is not a generalization as it is the same as the experiment type
        if not perform_generalizations:
            ex_types_for_generatilaztion = [ex_type]
        else:
            ex_types_for_generatilaztion = [
                "Architecture",
                "Seed",
                "Bootstrap",
            ]

        # Run the experiments for the generalization types
        for ex_generalization in ex_types_for_generatilaztion:
            logging.info(
                f"Running experiment for generalization type: {ex_generalization}"
            )

            model2_handles = []
            model2_times = []

            if "architecture" in ex_generalization.lower():
                hparams2_list = sample_architectures(
                    m_count_per_experiment, model_hp_pool
                )
                # add base hyperparameters to the list
                hparams2_list = [model_base_hp | h for h in hparams2_list]
            else:
                hparams2_list = [
                    model_base_hp | model_fixed_hparams
                ] * m_count_per_experiment

            if "bootstrap" in ex_generalization.lower():
                bootstrapM2 = sample_seeds(m_count_per_experiment)
            else:
                bootstrapM2 = [model_fixed_seed] * m_count_per_experiment

            if "seed" in ex_generalization.lower():
                seedM2 = sample_seeds(m_count_per_experiment)
            else:
                seedM2 = [model_fixed_seed] * m_count_per_experiment

            for model_2_index in range(m_count_per_experiment):
                seed2 = seedM2[model_2_index]
                hparams2 = hparams2_list[model_2_index]
                bootstrap2seed = bootstrapM2[model_2_index]

                t0 = time.time()
                model2, pred_proba2, pred_crisp2 = train_model_2(
                    X_train,
                    y_train,
                    ex_generalization,
                    model_type_to_use,
                    hparams2,
                    seed2,
                    bootstrap2seed,
                )
                time_model2 = time.time() - t0

                model2_handles.append(
                    (
                        f"Model2_{model_2_index}",
                        model2,
                        pred_proba2,
                        pred_crisp2,
                    )
                )
                model2_times.append(time_model2)

            logging.info(f"Finished training {m_count_per_experiment} M_2 models")

            for x in range(x_test_size):
                if x > X_test.shape[0] - 1:
                    logging.warning(
                        f"Test size {x} is larger than the test set size {X_test.shape[0]}. Skipping..."
                    )
                    continue

                # Obtain the test sample
                x_test_sample = X_test[x]
                y_test_sample = y_test[x]
                x_test_sample_pd = pd.DataFrame(
                    x_test_sample.reshape(1, -1), columns=X_test_pd.columns
                )

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

                    if base_cf is not None:
                        # Make sure it is flat
                        base_cf = base_cf.flatten()
                except Exception as e:
                    base_cf = None
                    logging.warning(f"BASE CF NOT FOUND, {e}")

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

                if just_base_cf:
                    _beta_confidences = [0.5]
                    _delta_robustnesses = [0.5]
                # If robx is used, then the beta_confidence and delta_robustness are not used so we set them to NaN
                elif robust_cf_method == "robx":
                    _beta_confidences = robx_taus  # use the taus for robx, but to simplify the code, we just assign it to beta_confidences variable
                    _delta_robustnesses = robx_variances  # use the variances for robx, but to simplify the code, we just assign it to delta_robustnesses variable
                else:  # otherwise, use the given values in config
                    _beta_confidences = beta_confidences
                    _delta_robustnesses = delta_robustnesses

                # Loop over the beta_confidence and delta_robustness values and the M_2 models
                for beta_confidence in _beta_confidences:
                    for delta_robustness in _delta_robustnesses:
                        for (
                            model2_name,
                            model2,
                            pred_proba2,
                            pred_crisp2,
                        ) in model2_handles:
                            # Start from calculating the validity of the base counterfactual
                            # Do this only once as it is the same for all M_2 models and all beta_confidence and delta_robustness
                            if first_flag:
                                if not check_is_none(base_cf):
                                    base_cf_validity_model2 = int(
                                        int(pred_crisp2(base_cf)[0]) == taget_class
                                    )
                                    base_cf_model1_pred_proba = pred_proba1(base_cf)
                                    base_cf_model1_pred_crisp = pred_crisp1(base_cf)
                                    base_cf_model2_pred_proba = pred_proba2(base_cf)
                                    base_cf_model2_pred_crisp = pred_crisp2(base_cf)
                                else:
                                    base_cf_validity_model2 = np.nan
                                    base_cf_model1_pred_proba = np.nan
                                    base_cf_model1_pred_crisp = np.nan
                                    base_cf_model2_pred_proba = np.nan
                                    base_cf_model2_pred_crisp = np.nan

                            base_cf_record = {
                                # Unique identifiers
                                "base_cf_method": base_cf_method,
                                "model_type_to_use": model_type_to_use,
                                "experiment_type": ex_type,
                                "dataset_name": dataset_name,
                                "just_base_cf": just_base_cf,
                                "k_mlps_in_B": k_mlps_in_B,
                                "fold_i": fold_i,
                                "experiment_generalization_type": ex_generalization,
                                "beta_confidence": beta_confidence,
                                "delta_robustness": delta_robustness,
                                "model2_name": model2_name,
                                # Utility data
                                "x_test_sample": [x_test_sample],
                                "y_test_sample": [y_test_sample],
                                "model1_pred_proba": pred_proba1_sample,
                                "model1_pred_crisp": pred_crisp1_sample,
                                "model2_pred_proba": pred_proba2(x_test_sample),
                                "model2_pred_crisp": pred_crisp2(x_test_sample),
                                # Base counterfactual data
                                "base_counterfactual": [base_cf],
                                "base_counterfactual_model1_pred_proba": base_cf_model1_pred_proba,
                                "base_counterfactual_model1_pred_crisp": base_cf_model1_pred_crisp,
                                "base_counterfactual_model2_pred_proba": base_cf_model2_pred_proba,
                                "base_counterfactual_model2_pred_crisp": base_cf_model2_pred_crisp,
                                "base_counterfactual_validity": base_metrics_model1[
                                    "validity"
                                ],
                                "base_counterfactual_proximityL1": base_metrics_model1[
                                    "proximityL1"
                                ],
                                "base_counterfactual_proximityL2": base_metrics_model1[
                                    "proximityL2"
                                ],
                                "base_counterfactual_plausibility": base_metrics_model1[
                                    "plausibility"
                                ],
                                "base_counterfactual_discriminative_power": base_metrics_model1[
                                    "dpow"
                                ],
                                "base_counterfactual_validity_model2": base_cf_validity_model2,
                                "base_counterfactual_time": time_base_cf,
                            }

                            # Obtain the robust counterfactual
                            # Check if the robustness is achievable and if the base counterfactual is not None
                            achievable = is_robustness_achievable_for_params(
                                k_mlps_in_B,
                                beta_confidence,
                                delta_robustness,
                            )
                            isNone = check_is_none(base_cf)
                            do_we_want_to_search = (
                                achievable
                                or not achievable
                                and robust_cf_method == "robx"
                            )
                            search_robust = (
                                do_we_want_to_search and not isNone and not just_base_cf
                            )

                            if search_robust:
                                t0 = time.time()
                                if robust_cf_method == "betarob":
                                    robust_cf, artifact_dict = betarce_generate(
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
                                        seed=global_random_state,
                                    )
                                elif robust_cf_method == "robx":
                                    robust_cf, _ = robx_algorithm(
                                        X_train=X_train,
                                        predict_class_proba_fn=pred_proba1,
                                        start_counterfactual=base_cf,
                                        tau=beta_confidence,  # This is the tau parameter in the robx algorithm, just reusing the beta_confidence for simplicity
                                        variance=delta_robustness,
                                        N=robx_N,
                                    )
                                    artifact_dict = {}
                                else:
                                    raise ValueError("Unknown robust cf method")
                                time_robust_cf = time.time() - t0
                            else:
                                robust_cf = None
                                artifact_dict = {
                                    "start_sample_passes_test": np.nan,
                                    "counterfactual_does_not_pass_test": np.nan,
                                    "counterfactual_does_not_have_target_class": np.nan,
                                    "counterfactual_is_nan": np.nan,
                                    "highest_delta": np.nan,
                                }
                                time_robust_cf = np.nan

                            # Calculate the metrics
                            robust_metrics_model1 = calculate_metrics(
                                cf=robust_cf,
                                cf_desired_class=taget_class,
                                x=x_test_sample,
                                y_train=y_train,
                                nearest_neighbors_model=nearest_neighbors_model,
                                predict_fn_crisp=pred_crisp1,
                            )

                            if not check_is_none(robust_cf):
                                robust_cf_validity_model2 = int(
                                    int(pred_crisp2(robust_cf)[0]) == taget_class
                                )
                                robust_cf_L1_distance_from_base_cf = np.sum(
                                    np.abs(robust_cf - base_cf)
                                )
                                robust_cf_L2_distance_from_base_cf = np.sum(
                                    np.square(robust_cf - base_cf)
                                )
                                robust_cf_model1_pred_proba = pred_proba1(robust_cf)
                                robust_cf_model1_pred_crisp = pred_crisp1(robust_cf)
                                robust_cf_model2_pred_proba = pred_proba2(robust_cf)
                                robust_cf_model2_pred_crisp = pred_crisp2(robust_cf)
                            else:
                                robust_cf_validity_model2 = np.nan
                                robust_cf_L1_distance_from_base_cf = np.nan
                                robust_cf_L2_distance_from_base_cf = np.nan
                                robust_cf_model1_pred_proba = np.nan
                                robust_cf_model1_pred_crisp = np.nan
                                robust_cf_model2_pred_proba = np.nan
                                robust_cf_model2_pred_crisp = np.nan

                            # Store the results in the frame
                            robust_cf_record = {
                                # Robust counterfactual data
                                "robust_counterfactual": [robust_cf],
                                "robust_counterfactual_model1_pred_proba": robust_cf_model1_pred_proba,
                                "robust_counterfactual_model1_pred_crisp": robust_cf_model1_pred_crisp,
                                "robust_counterfactual_model2_pred_proba": robust_cf_model2_pred_proba,
                                "robust_counterfactual_model2_pred_crisp": robust_cf_model2_pred_crisp,
                                "robust_counterfactual_validity": robust_metrics_model1[
                                    "validity"
                                ],
                                "robust_counterfactual_proximityL1": robust_metrics_model1[
                                    "proximityL1"
                                ],
                                "robust_counterfactual_proximityL2": robust_metrics_model1[
                                    "proximityL2"
                                ],
                                "robust_counterfactual_plausibility": robust_metrics_model1[
                                    "plausibility"
                                ],
                                "robust_counterfactual_discriminative_power": robust_metrics_model1[
                                    "dpow"
                                ],
                                "robust_counterfactual_validity_model2": robust_cf_validity_model2,
                                "robust_counterfactual_L1_distance_from_base_cf": robust_cf_L1_distance_from_base_cf,
                                "robust_counterfactual_L2_distance_from_base_cf": robust_cf_L2_distance_from_base_cf,
                                "robust_counterfactual_time": time_robust_cf,
                            }
                            # Add artifact_dict to the record
                            record = {
                                **base_cf_record,
                                **robust_cf_record,
                                **artifact_dict,
                            }
                            record = pd.DataFrame(record, index=[0])
                            results_df = pd.concat(
                                [results_df, record], ignore_index=True
                            )

                            # Save the results every n iterations
                            if (
                                global_iteration % save_every_n_iterations == 0
                                and global_iteration > 0
                            ):
                                results_df.to_feather(
                                    f"{results_df_dir}/{global_iteration}_results.feather"
                                )
                                # results_df.to_csv(
                                #     f"{results_df_dir}/{global_iteration}_results.csv"
                                # )
                                cols = results_df.columns

                                # Clear the results_df to save memory and speed up the process
                                del results_df
                                results_df = pd.DataFrame(columns=cols)

                            global_iteration += 1
                            tqdm_pbar.update(1)

                        first_flag = False

    # Final save
    results_df.to_feather(f"{results_df_dir}/{global_iteration}_results.feather")
    # results_df.to_csv(f"{results_df_dir}/{global_iteration}_results.csv")
    # results_df = pd.DataFrame(columns=results_df.columns)

    # Progress bar close
    tqdm_pbar.close()
    logging.info("Finished all experiments")


if __name__ == "__main__":
    experiment()
