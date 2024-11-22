import yaml
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from datasets import Dataset, DatasetPreprocessor
from classifiers.mlpclassifier import (
    MLPClassifier,
    train_neural_network,
    train_K_mlps_in_parallel,
)
from classifiers.rfclassifier import (
    RFClassifier,
    train_random_forest,
    train_K_rfs_in_parallel,
)
from classifiers.dtclassifier import (
    DecisionTree,
    train_decision_tree,
    train_K_dts_in_parallel,
)
from classifiers.lgbmclassifier import (
    LGBMClassifier,
    train_lgbm,
    train_K_LGBMS_in_parallel,
)
from classifiers.baseclassifier import BaseClassifier
from classifiers.utils import bootstrap_data
from explainers import (
    DiceExplainer,
    GrowingSpheresExplainer,
    BaseExplainer,
    CarlaExplainer,
    RBRExplainer,
)
from betarob import BetaRob


def get_config(path: str = "./config.yml") -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train_model(X_train, y_train, model_type: str, seed: int, hparams: dict) -> tuple:
    if model_type == "neural_network":
        m, pp, pc = train_neural_network(X_train, y_train, seed, hparams)
    elif model_type == "random_forest":
        m, pp, pc = train_random_forest(X_train, y_train, seed, hparams)
    elif model_type == "decision_tree":
        m, pp, pc = train_decision_tree(X_train, y_train, seed, hparams)
    elif model_type == "lightgbm":
        m, pp, pc = train_lgbm(X_train, y_train, seed, hparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return m, pp, pc


def train_B(
    ex_type: str,
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
        case "neural_network":
            results = train_K_mlps_in_parallel(
                X_train=X_train,
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
            models = [
                model
                for partial_results in results
                for model in partial_results["models"]
            ]
        case "decision_tree":
            results = train_K_dts_in_parallel(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hparamsB=hparamsB,
                bootstrapB=bootstrapB,
                seedB=seedB,
                hparams_base=model_base_hyperparameters,
                K=k_mlps_in_B,
                n_jobs=n_jobs,
            )
            models = [
                model
                for partial_results in results
                for model in partial_results["models"]
            ]
        case "lightgbm":
            results = train_K_LGBMS_in_parallel(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hparamsB=hparamsB,
                bootstrapB=bootstrapB,
                seedB=seedB,
                hparams_base=model_base_hyperparameters,
                K=k_mlps_in_B,
                n_jobs=n_jobs,
            )
            models = [
                model
                for partial_results in results
                for model in partial_results["models"]
            ]
        case _:
            raise ValueError("Unknown model type. Cannot train B models.")

    return models


def train_model_2(
    X_train,
    y_train,
    ex_type: str,
    model_type: str,
    hparams: dict,
    seed: int,
    bootstrap_seed: int,
) -> tuple:
    # Perform bootrap with the bootstrap seed
    if "bootstrap" in ex_type.lower():
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
        case "dice":
            X_train_w_target = X_train.copy()
            X_train_w_target[dataset_preprocessor.target_column] = y_train
            explainer = DiceExplainer(
                dataset=X_train_w_target,
                model=model,
                outcome_name=dataset_preprocessor.target_column,
                continous_features=dataset_preprocessor.continuous_columns,
            )
            explainer.prep(dice_method="random", feature_encoding=None)
        case "gs":
            explainer = GrowingSpheresExplainer(
                keys_mutable=dataset_preprocessor.X_train.columns.tolist(),
                keys_immutable=[],
                feature_order=dataset_preprocessor.X_train.columns.tolist(),
                binary_cols=dataset_preprocessor.transformed_features.tolist(),
                continous_cols=dataset_preprocessor.continuous_columns,
                pred_fn_crisp=predict_fn_1_crisp,
                target_proba=hparams["target_proba"],
                max_iter=hparams["max_iter"],
                n_search_samples=hparams["n_search_samples"],
                p_norm=hparams["p_norm"],
                step=hparams["step"],
            )
            explainer.prep()
        case "face" | "roar" | "clue":
            X_train_w_target = X_train.copy()
            X_train_w_target[dataset_preprocessor.target_column] = y_train
            explainer = CarlaExplainer(
                train_dataset=X_train_w_target,
                explained_model=model,
                continous_columns=dataset_preprocessor.continuous_columns,
                categorical_columns=dataset_preprocessor.categorical_columns,
                target_feature_name=dataset_preprocessor.target_column,
                nonactionable_columns=[],
                columns_order_ohe=X_train.columns.tolist(),
            )
            explainer.prep(method_to_use=base_cf_method)
        case "rbr":
            explainer = RBRExplainer(X_train.copy(), model)
            explainer.prep()
        case _:
            raise ValueError(
                "base_cf_method name not recognized. Make sure to set it in config"
            )

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
    elif isinstance(base_explainer, CarlaExplainer):
        return base_explainer.generate(instance)
    elif isinstance(base_explainer, RBRExplainer):
        return base_explainer.generate(instance)
    else:
        raise ValueError(
            "base_explainer must be either a DiceExplainer or a GrowingSpheresExplainer"
        )


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
        seed=seed,
    )

    beta_explainer.prep()

    robust_cf, artifact_dict = beta_explainer.generate(
        start_instance=start_instance,
        target_class=target_class,
        delta_target=delta_target,
        beta_confidence=beta_confidence,
    )

    return robust_cf, artifact_dict


def calculate_metrics(
    cf: np.ndarray,
    cf_desired_class: int,
    x: np.ndarray,
    y_train: np.ndarray,
    nearest_neighbors_model: NearestNeighbors,
    predict_fn_crisp: callable,
    dpow_neighbours: int = 15,
    plausibility_neighbours: int = 15,
) -> dict[str, float | int]:
    """
    Calculates the metrics for a counterfactual example.
    """

    if check_is_none(cf):
        return {
            "validity": np.nan,
            "proximityL1": np.nan,
            "proximityL2": np.nan,
            "dpow": np.nan,
            "plausibility": np.nan,
        }

    cf_label = predict_fn_crisp(cf)[0]

    # Validity
    validity = int(int(cf_label) == cf_desired_class)

    # Proximity L1
    proximityL1 = np.sum(np.abs(x - cf))

    # Proximity L2
    proximityL2 = np.sqrt(np.sum(np.square(x - cf)))

    # Discriminative Power (fraction of neighbors with the same label as the counterfactual)
    neigh_indices = nearest_neighbors_model.kneighbors(
        cf.reshape(1, -1), return_distance=False, n_neighbors=dpow_neighbours
    )
    neigh_labels = y_train[neigh_indices[0]]
    dpow = np.sum(neigh_labels == cf_label) / len(
        neigh_labels
    )  # The fraction of neighbors with the same label as the counterfactual

    # Plausibility (average distance to the 50 nearest neighbors in the training data)
    neigh_dist, _ = nearest_neighbors_model.kneighbors(
        cf.reshape(1, -1), return_distance=True, n_neighbors=plausibility_neighbours
    )
    plausibility = np.mean(neigh_dist[0])

    return {
        "validity": validity,
        "proximityL1": proximityL1,
        "proximityL2": proximityL2,
        "dpow": dpow,
        "plausibility": plausibility,
    }


def check_is_none(to_check: object) -> bool:
    """Check if the object is None or np.nan or has any NaN values."""
    if to_check is None or to_check is np.nan:
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


def is_robustness_achievable_for_params(
    k: int, beta_confidence: float, delta_robustness: float
) -> bool:
    """
    Check if with the given parameters the robustness is achievable.
    """
    lb, _ = stats.beta.interval(beta_confidence, 0.5 + k, 0.5)
    return lb > delta_robustness


def sample_architectures(n: int, hparams: dict) -> list[dict]:
    """
    Sample n architectures from the hyperparameters pool
    """
    architectures = []
    for _ in range(n):
        architecture = {}
        for _param, _options in hparams.items():
            if isinstance(_options, list):
                architecture[_param] = np.random.choice(_options)
            elif isinstance(_options, dict):
                lower = _options["lower"]
                upper = _options["upper"]
                # Check if the lower and upper are integers
                if isinstance(lower, int) and isinstance(upper, int):
                    lower = int(lower)
                    upper = int(upper)
                    architecture[_param] = np.random.randint(lower, upper + 1)
                # Otherwise, they are floats
                else:
                    freq = _options["freq"]
                    lower, upper, freq = float(lower), float(upper), int(freq)
                    architecture[_param] = np.random.uniform(lower, upper, freq)
            else:
                raise ValueError("Unknown hyperparameter type", _options, "for", _param)
        architectures.append(architecture)
    return architectures


def sample_seeds(n: int) -> list[int]:
    """
    Sample n seeds for the bootstrap
    """
    seeds = np.random.choice(1000, n, replace=False)
    return seeds
