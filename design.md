Parameters:
1. EX_types = [ "Bootstrap", "Architecture", "Bootstrap-Architecture", "Seed", "Seed-Bootstrap", "Seed-Architecture", "Seed-Bootstrap-Architecture" ]  
1. Datasets = [ "fico", "wine", "breast_cancer"]
1. Beta_confidences = [ 0.8, 0.9, 0.95, 0.99 ]
1. Delta_robustness = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
1. X_test_size = 100
1. M_count_per_experiment = 15
1. Cross_validation_folds = 5

Scheme: 
HYPERPARAMETERS:
- Base_counterfactual_method = "GrowingSpheres"
- Model_type = "NeuralNetwork"
- Model_hyperparameters_pool = { "hidden_layers": [2,3], "neurons_per_layer": [64, 128], "activation": ["relu"], "optimizer": ["adam", "sgd"], "loss": ["mean_squared_error"] }
- Model_fixed_hyperparameters = {"hidden_layers": 3, "neurons_per_layer": 128, "activation": "relu", "optimizer": "adam", "loss": "mean_squared_error}
- Model_fixed_seed = 42
- metrics_to_calculate = ["validity", "proximityL1", "proximityL2" "plausibility", "dpow"]


cf_metrics = [f"{base_or_robust}_{model_1_or_2}_{metric_name}" for base_or_robust in ["base", "robust"] for model_1_or_2 in [1, 2] for metric_name in metrics_to_calculate]

results_template = pd.DataFrame(columns=["Base_cf_method", "Model_type", "Dataset", "EX_type", "CVFold", "Ex_generalization" "M_idx", "Beta", "Delta", "CF", "CF_robust", **cf_metrics])



1. For each Dataset
    1. For each EX_type
        1. For each cross-validation fold
            - Create `X_train`, `X_test` from the fold  
            - Train `M_1` on `X_train`, `Model_fixed_seed` and `Model_fixed_hyperparameters`

            - For each model `b` in `B` estimators
                - if "Bootstrap" in EX_type:
                    - `X_train_1` <- sample bootstrap `X_train`
                - if "Architecture" in EX_type:
                    - `M_1 hparams` <- sample hyperparameters from `Model_hyperparameters_pool`
                - if "Seed" in EX_type:
                    - `M_1 seed` <- sample seed
                - Train `b` on `X_train_1`, `M_1 seed` and `M_1 hparams`

            1. For each `ex_generalization` in EX_types: 
                1. For each `m` in 1 ... M_count_per_experiment:  
                    - if "Architecture" in ex_generalization:
                        - `X_train_2` <- `X_train_1` 
                        - `M_2 seed` <- `M_1 seed`
                        - `M_2 hparams` <- sample hyperparameters from `Model_hyperparameters_pool`
                    - if "Seed" in ex_generalization:
                        - `X_train_2` <- `X_train_1`
                        - `M_2 seed` <- sample new seed 
                        - `M_2 hparams` <- `M_1 hparams`
                    - if "Bootstrap" in ex_generalization:
                        - `X_train_2` <- sample new bootstrap `X_train_1`
                        - `M_2 seed` <- `M_1 seed`
                        - `M_2 hparams` <- `M_1 hparams`
                    - Train `M_2` on `X_train_2`, `M_2 seed` and `M_2 hparams`
                    1. For each `x` in random.permutation(`X_test`):  
                        - Obtain base counterfactual `cf` for `x` using `M_1`
                        1. For each beta in Beta_confidences:
                        1. For each delta in Delta_robustness:
                                - Obtain robust counterfactual `cf_robust` for `x` using `M_2` and `B` with `beta` and `delta`
                                - Store `cf_robust` and `cf` in the results
                                - Calculate metrics for `cf_robust` and `cf` for `M_1` and `M_2` and store them in the results frame
                                    Use template: f"{base_or_robust}_{model_1_or_2}_{metric_name}"


