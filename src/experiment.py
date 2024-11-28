import os
import time
import logging
import yaml
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
from itertools import product
import traceback
from tqdm import tqdm
import warnings

from .datasets import DatasetPreprocessor, Dataset
from .experiments_utils import (
    get_B,
    train_B,
    prepare_base_counterfactual_explainer,
    check_is_none,
    calculate_metrics,
    is_robustness_achievable_for_params,
)
from .classifiers.baseclassifier import BaseClassifier
from .explainers.base import BaseExplainer
from .explainers.posthoc import BetaRob, RobX, PostHocExplainer


# Suppress specific warning category
warnings.filterwarnings("ignore", category=UserWarning)


class Experiment:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self._setup()

        if self.cfg_exp["pretrain"]:
            self.pretrain_classifiers()

        self.results_list = list()
        self.global_iter = 0

    def run(self) -> None:
        """
        Run the experiment
        """
        base_explainers = self.cfg_exp["base_explainers"]
        e2e_explainers = self.cfg_exp["e2e_explainers"]
        posthoc_explainers = self.cfg_exp["posthoc_explainers"]

        exps = {}
        if e2e_explainers is not None:
            exps = exps | {k: "e2e" for k in e2e_explainers}
        if base_explainers is not None:
            exps = exps | {k: "base" for k in base_explainers}

        all_combinations = []
        for dataset_name in self.cfg_exp["datasets"]:
            for fold in range(self.cfg_exp["cross_validation_folds"]):
                for classifier in self.cfg_exp["classifiers"]:
                    for exp_type in self.cfg_exp["ex_types"]:
                        for explainer in exps.items():
                            all_combinations.append(
                                (dataset_name, fold, classifier, exp_type, explainer)
                            )

        total_iters = self.calculate_total_iterations(all_combinations)
        logging.info(f"Total iterations: {total_iters}")

        self.pbar = tqdm(total=total_iters, desc="Global Iterations")

        for combination in all_combinations:
            dataset_name, fold, classifier, exp_type, explainer = combination

            is_base = explainer[1] == "base"
            is_e2e = explainer[1] == "e2e"
            explainer_name = explainer[0]
            class_thresh = (
                self.cfg_mod_hp[classifier]["model_base_hyperparameters"][
                    "classification_threshold"
                ],
            )

            # Get dataset
            dataset = Dataset(dataset_name, self.cfg_gen["random_seed"])

            # Get the preprocessor
            preprocessor = DatasetPreprocessor(
                dataset=dataset,
                cross_validation_folds=self.cfg_exp["cross_validation_folds"],
                fold_idx=fold,
                random_state=self.cfg_gen["random_seed"],
                one_hot=True,
            )

            # Get the data
            X_train_pd, X_test_pd, y_train, y_test = preprocessor.get_data()

            # Convert to numpy
            X_train, X_test = [x.to_numpy() for x in (X_train_pd, X_test_pd)]

            # Random shuffle the data
            shuffle_indices = np.random.permutation(X_train.shape[0])
            X_train, y_train = (
                X_train[shuffle_indices],
                y_train[shuffle_indices],
            )

            # Prepare the nearest neighbors model for the metrics
            self.nearest_neighbors_model = NearestNeighbors(n_neighbors=20, n_jobs=1)
            self.nearest_neighbors_model.fit(X_train)

            # Get x_test_size M1 models
            models_m1 = self.sample_classifiers(
                1,
                exp_type,
                classifier,
                dataset_name,
                fold,
            )

            # Get m2_count M2 models
            models_m2 = self.sample_classifiers(
                self.cfg_exp["m2_count"],
                exp_type,
                classifier,
                dataset_name,
                fold,
            )

            if is_e2e:
                e2e_combs = self.get_e2e_combinations(explainer_name, self.cfg)
            else:
                e2e_combs = [({}, {})]

            logging.info(f"Combination: {combination}")

            for base_hp, base_hp_to_save in e2e_combs:
                # Get the base explainer
                if not is_e2e:
                    base_hp = self.cfg[explainer_name]

                model1: BaseClassifier = models_m1[0]
                base_explainer = self.get_base_explainer(
                    X_train_pd, y_train, explainer_name, base_hp, model1, preprocessor
                )

                # select a subset of the test set
                x_test_size = self.cfg_exp["x_test_size"]
                indices = np.random.choice(X_test.shape[0], x_test_size, replace=False)

                for _i, idx in enumerate(indices):
                    for m2 in models_m2:
                        # logging.info(f"Global iteration: {self.global_iter}")

                        # Obtain the test sample
                        x_np = X_test[idx]
                        y = y_test[idx]
                        x = pd.DataFrame(x_np.reshape(1, -1), columns=X_test_pd.columns)

                        # Obtain the base cf
                        t0 = time.time()
                        try:
                            b_cf = base_explainer.generate(x)
                            if check_is_none(b_cf):
                                b_cf = None
                            else:
                                b_cf = b_cf.flatten()
                        except Exception as e:
                            logging.error(f"Base CF not found: {e}")
                            traceback.print_exc(file=sys.stderr)
                            b_cf = None
                        b_cf_time = time.time() - t0

                        base_record = self.get_base_record(combination)
                        if not check_is_none(b_cf):
                            base_record = base_record | self.get_m1_m2_records(
                                model1,
                                m2,
                                x,
                                y,
                                y_train,
                                b_cf,
                                "base",
                                b_cf_time,
                            )
                        base_record["is_base"] = is_base
                        base_record["is_e2e"] = is_e2e

                        if is_e2e:
                            base_record = base_record | base_hp_to_save

                        self.add_to_results(deepcopy(base_record))

                        # Skip the posthoc explainer if the base explainer is None
                        # or if it is an end-to-end explainer
                        if is_e2e or check_is_none(b_cf):
                            self.global_iter += 1
                            self.pbar.update(1)
                            continue

                        # Get the posthoc explainer hparams
                        for ph_explainer_name in posthoc_explainers:
                            self.robust_part(
                                ph_explainer_name,
                                combination,
                                preprocessor,
                                model1,
                                m2,
                                x,
                                y,
                                y_train,
                                b_cf,
                                class_thresh,
                                base_record,
                            )

            self.save_results()
        self.pbar.close()

    def robust_part(
        self,
        ph_explainer_name,
        combination,
        preprocessor,
        model1,
        m2,
        x,
        y,
        y_train,
        b_cf,
        class_thresh,
        base_record,
    ):
        # Get the posthoc explainer
        ph_explainer = self.get_posthoc_explainer(
            ph_explainer_name,
            preprocessor,
            model1,
            combination,
            class_thresh,
        )

        ph_combs = self.generate_posthoc_hp_combinations(ph_explainer_name)
        ph_fixed = self.cfg[ph_explainer_name]["fixed_hps"]

        for ph_comb in ph_combs:
            if (
                ph_explainer_name == "betarob"
                and not is_robustness_achievable_for_params(**ph_comb)
            ):
                logging.info("Skip BetaRob -- unachievable robustness")
                continue

            t0 = time.time()
            try:
                rb_cf, artifacts_dict = ph_explainer.generate(
                    start_instance=b_cf,
                    target_class=1 - model1.predict_crisp(x)[0],
                    **(ph_comb | ph_fixed),
                )
                if check_is_none(rb_cf):
                    rb_cf = None
                else:
                    rb_cf = rb_cf.flatten()
            except Exception as e:
                logging.error(f"Posthoc CF not found: {e}")
                traceback.print_exc(file=sys.stderr)
                rb_cf = None
            rb_cf_time = time.time() - t0

            if not check_is_none(rb_cf):
                rb_record = base_record | self.get_m1_m2_records(
                    model1,
                    m2,
                    x,
                    y,
                    y_train,
                    rb_cf,
                    "robust",
                    rb_cf_time,
                )
            rb_record["posthoc_explainer"] = ph_explainer_name
            for k, v in ph_comb.items():
                rb_record[k] = v

            L1_dist_to_base = np.sum(np.abs(b_cf - rb_cf))
            L2_dist_to_base = np.sqrt(np.sum(np.square(b_cf - rb_cf)))
            rb_record["robust_counterfactual_L1_distance_from_base"] = L1_dist_to_base
            rb_record["robust_counterfactual_L2_distance_from_base"] = L2_dist_to_base

            # Add the artifacts to the record
            for k, v in artifacts_dict.items():
                rb_record[k] = v

            self.add_to_results(deepcopy(rb_record))

            # Iterate the global iteration
            self.global_iter += 1
            self.pbar.update(1)

    def calculate_total_iterations(self, all_combs) -> int:
        c = len(all_combs)
        x_test_size = self.cfg_exp["x_test_size"]
        posthoc_explainers = self.cfg_exp["posthoc_explainers"]

        hp_per_posthoc = 0
        for ph_explainer_name in posthoc_explainers:
            hp_per_posthoc += len(
                self.generate_posthoc_hp_combinations(ph_explainer_name)
            )

        m2_count = self.cfg_exp["m2_count"]

        hp_per_e2e = 0
        if self.cfg_exp["e2e_explainers"] is not None:
            for explainer in self.cfg_exp["e2e_explainers"]:
                hp_per_e2e += len(self.get_e2e_combinations(explainer, self.cfg))

        return c * x_test_size * m2_count * (hp_per_posthoc + hp_per_e2e)

    def add_to_results(self, record: dict) -> None:
        self.results_list.append(record)

        if len(self.results_list) % self.cfg_exp["save_every"] == 0:
            self.save_results()

    def save_results(self) -> None:
        if len(self.results_list) == 0:
            return

        df = pd.DataFrame(self.results_list)
        format = self.cfg_gen["save_format"]
        path = os.path.join(
            self.cfg_gen["result_path"],
            "results",
            f"results_{self.global_iter}.{format}",
        )

        if format == "csv":
            df.to_csv(path)
        elif format == "feather":
            df.to_feather(path)
        else:
            raise ValueError(f"Unknown save format: {format}")

        self.results_list = []

    def get_base_record(self, combination: tuple) -> dict:
        record = {
            "dataset_name": combination[0],
            "fold_i": combination[1],
            "model_type_to_use": combination[2],
            "experiment_type": combination[3],
            "base_cf_method": combination[4][0],
        }
        return record

    def get_m1_m2_records(
        self,
        m1: BaseClassifier,
        m2: BaseClassifier,
        x: pd.DataFrame,
        y: int,
        y_train: np.ndarray,
        cf: np.ndarray,
        cf_prefix: str,
        time_taken: float,
    ) -> dict:
        m1_pred_proba = m1.predict_proba(x)[0]
        m2_pred_proba = m2.predict_proba(x)[0]
        m1_pred_crisp = m1.predict_crisp(x)[0]
        m2_pred_crisp = m2.predict_crisp(x)[0]

        cf_m1_pred_proba = m1.predict_proba(cf)[0]
        cf_m2_pred_proba = m2.predict_proba(cf)[0]
        cf_m1_pred_crisp = m1.predict_crisp(cf)[0]
        cf_m2_pred_crisp = m2.predict_crisp(cf)[0]

        cf_prefix = cf_prefix + "_"

        record = {
            "x_test_sample": [x.values.flatten()],
            "y_test_sample": y,
            "model1_pred_proba": m1_pred_proba,
            "model1_pred_crisp": m1_pred_crisp,
            "model2_pred_proba": m2_pred_proba,
            "model2_pred_crisp": m2_pred_crisp,
            f"{cf_prefix}counterfactual": [cf],
            f"{cf_prefix}counterfactual_model1_pred_proba": cf_m1_pred_proba,
            f"{cf_prefix}counterfactual_model1_pred_crisp": cf_m1_pred_crisp,
            f"{cf_prefix}counterfactual_model2_pred_proba": cf_m2_pred_proba,
            f"{cf_prefix}counterfactual_model2_pred_crisp": cf_m2_pred_crisp,
        }

        cf_target_class = 1 - m1_pred_crisp
        cf_validity_m2 = cf_target_class == cf_m2_pred_crisp

        metrics_m1 = calculate_metrics(
            cf=cf,
            cf_desired_class=cf_target_class,
            x=x.values.flatten(),
            y_train=y_train,
            nearest_neighbors_model=self.nearest_neighbors_model,
            predict_fn_crisp=m1.predict_crisp,
        )

        mets = {
            f"{cf_prefix}counterfactual_validity": metrics_m1["validity"],
            f"{cf_prefix}counterfactual_validity_model2": cf_validity_m2,
            f"{cf_prefix}counterfactual_proximityL1": metrics_m1["proximityL1"],
            f"{cf_prefix}counterfactual_proximityL2": metrics_m1["proximityL2"],
            f"{cf_prefix}counterfactual_plausibility": metrics_m1["plausibility"],
            f"{cf_prefix}counterfactual_discriminative_power": metrics_m1["dpow"],
            f"{cf_prefix}counterfactual_time": time_taken,
        }

        record = record | mets

        return record

    def get_base_explainer(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        base_explainer: str,
        explainer_hp: dict,
        model: BaseClassifier,
        preprocessor: DatasetPreprocessor,
    ) -> BaseExplainer:
        explainer = prepare_base_counterfactual_explainer(
            base_cf_method=base_explainer,
            hparams=explainer_hp,
            model=model,
            X_train=X_train,
            y_train=y_train,
            dataset_preprocessor=preprocessor,
            predict_fn_1_crisp=model.predict_crisp,
        )
        return explainer

    def get_posthoc_explainer(
        self,
        name: str,
        preprocessor: DatasetPreprocessor,
        model: BaseClassifier,
        combination: tuple,
        class_threshold: float,
    ) -> PostHocExplainer:
        if name == "betarob":
            k_estimators = self.cfg[name]["pool_hps"]["k"]
            estimators = self.sample_classifiers(
                k_estimators,
                combination[3],
                combination[2],
                combination[0],
                combination[1],
            )

            estimators_crisp = [est.predict_crisp for est in estimators]

            ph_explainer = BetaRob(
                dataset=preprocessor.dataset,
                preprocessor=preprocessor,
                pred_fn_crisp=model.predict_crisp,
                pred_fn_proba=model.predict_proba,
                estimators_crisp=estimators_crisp,
                grow_sphere_hparams=self.cfg[name]["fixed_hps"]["gs"],
                classification_threshold=class_threshold,
                seed=self.cfg_gen["random_seed"],
            )

            ph_explainer.prep()
        elif name == "robx":
            ph_explainer = RobX(
                X_train=preprocessor.X_train,
                predict_proba_fn=model.predict_proba,
            )

            ph_explainer.prep()
        else:
            raise ValueError(f"Unknown posthoc explainer: {name}")

        return ph_explainer

    def generate_posthoc_hp_combinations(self, posthoc_explainer: str) -> list[dict]:
        if posthoc_explainer == "betarob":
            pool_hps = self.cfg[posthoc_explainer]["pool_hps"]

        elif posthoc_explainer == "robx":
            pool_hps = self.cfg[posthoc_explainer]["pool_hps"]
        else:
            raise ValueError(f"Unknown posthoc explainer: {posthoc_explainer}")

        # Do all the combinations outer product
        all_combinations = list(product(*pool_hps.values()))
        labels = list(pool_hps.keys())
        # insert labels and turn tuples into dicts
        all_combinations = [dict(zip(labels, comb)) for comb in all_combinations]
        return all_combinations

    def sample_classifiers(
        self, k: int, exp_type: str, classifier: str, dataset: str, fold: str | int
    ) -> list:
        """
        Sample k models from the specified experiment type, classifier, dataset and fold
        """
        path = os.path.join(
            self.cfg_gen["model_path"],
            dataset,
            exp_type,
            classifier,
            fold if isinstance(fold, str) else str(fold),
        )
        # discover the models
        filenames = os.listdir(path)

        np.random.seed(
            self.cfg_gen["random_seed"]
        )  # TODO: this is temporary to always get the same models

        # sample k models
        sampled_models = np.random.choice(filenames, k, replace=False)

        models = []
        for model in sampled_models:
            model_path = os.path.join(path, model)
            models.append(joblib.load(model_path))

        return models

    def pretrain_classifiers(self):
        """
        With the given configuration, pretrain the classifiers and save them to the disk
        """
        exp_types = self.cfg_exp["ex_types"]
        datasets = self.cfg_exp["datasets"]
        folds = self.cfg_exp["cross_validation_folds"]
        classifiers = self.cfg_exp["classifiers"]
        pretrainN = self.cfg_exp["n_pretrain"]

        all_combinations = []
        for dataset_name in datasets:
            for fold in range(folds):
                for classifier in classifiers:
                    for exp_type in exp_types:
                        all_combinations.append(
                            (dataset_name, fold, classifier, exp_type)
                        )

        for dataset_name, fold, classifier, exp_type in all_combinations:
            # Get dataset
            dataset = Dataset(dataset_name, self.cfg_gen["random_seed"])

            preprocessor = DatasetPreprocessor(
                dataset=dataset,
                cross_validation_folds=self.cfg_exp["cross_validation_folds"],
                fold_idx=fold,
                random_state=self.cfg_gen["random_seed"],
                one_hot=True,
            )

            X_train_pd, X_test_pd, y_train, y_test = preprocessor.get_data()

            # Convert to numpy
            X_train, X_test = [x.to_numpy() for x in (X_train_pd, X_test_pd)]
            # Random shuffle the data
            shuffle_indices = np.random.permutation(X_train.shape[0])
            X_train, y_train = (
                X_train[shuffle_indices],
                y_train[shuffle_indices],
            )

            model_fixed_seed = self.cfg_mod_hp[classifier]["model_fixed_seed"]
            model_fixed_hp = self.cfg_mod_hp[classifier]["model_fixed_hyperparameters"]
            model_hp_pool = self.cfg_mod_hp[classifier]["model_hyperparameters_pool"]
            model_base_hp = self.cfg_mod_hp[classifier]["model_base_hyperparameters"]

            hparams_pool, seed_pool, bootstrap_pool = get_B(
                exp_type,
                pretrainN,
                model_fixed_hp,
                model_hp_pool,
                model_fixed_seed,
            )

            print(seed_pool)
            print(hparams_pool)
            print(bootstrap_pool)

            models_pool = train_B(
                model_type_to_use=classifier,
                model_base_hyperparameters=model_base_hp,
                seedB=seed_pool,
                hparamsB=hparams_pool,
                bootstrapB=bootstrap_pool,
                k_mlps_in_B=pretrainN,
                n_jobs=model_base_hp["n_jobs"] if "n_jobs" in model_base_hp else 1,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            # Folder structure
            # dataset
            #    exp_type
            #       classifier
            #           fold
            # filenames follow indices of the models in the pool
            # save the models
            dataset_dir = os.path.join(self.cfg_gen["model_path"], dataset_name)
            exp_type_dir = os.path.join(dataset_dir, exp_type)
            classifier_dir = os.path.join(exp_type_dir, classifier)
            fold_dir = os.path.join(classifier_dir, str(fold))
            os.makedirs(fold_dir, exist_ok=True)

            for idx, model in enumerate(models_pool):
                model_path = os.path.join(fold_dir, f"model_{idx}.joblib")
                joblib.dump(model, model_path)

    def get_e2e_combinations(self, base_cf_method: str, config: dict) -> list:
        if base_cf_method == "roar":
            hparams = config["roar"]
            all_combinations = []
            for dmx in hparams["delta_max"]:
                for lr in hparams["lr"]:
                    for norm in hparams["norm"]:
                        _base = deepcopy(hparams)
                        _base["delta_max"] = dmx
                        _base["lr"] = lr
                        _base["norm"] = norm
                        hparams_to_save = _base.copy()
                        all_combinations.append((_base, hparams_to_save))
        elif base_cf_method == "rbr":
            hparams: dict = config["rbr"]
            all_combinations = []
            for pr in hparams["perturb_radius"]["synthesis"]:
                for dp in hparams["ec"]["rbr_params"]["delta_plus"]:
                    for s in hparams["ec"]["rbr_params"]["sigma"]:
                        _base = deepcopy(hparams)
                        _base["perturb_radius"]["synthesis"] = pr
                        _base["ec"]["rbr_params"]["delta_plus"] = dp
                        _base["ec"]["rbr_params"]["sigma"] = s
                        hparams_to_save = {
                            "perturb_radius": pr,
                            "delta_plus": dp,
                            "sigma": s,
                        }
                        all_combinations.append((_base, hparams_to_save))
        elif base_cf_method == "face":
            hparams = config["face"]
            all_combinations = []
            for f in hparams["fraction"]:
                for m in hparams["mode"]:
                    _base = deepcopy(hparams)
                    _base["fraction"] = f
                    _base["mode"] = m
                    hp_to_save = _base.copy()
                    all_combinations.append((_base, hp_to_save))
        elif base_cf_method == "gs":
            hparams = config["gs"]
            all_combinations = [(hparams, hparams)]
        elif base_cf_method == "dice":
            hparams = config["dice"]
            all_combinations = []
            for pw in hparams["proximity_weight"]:
                for dw in hparams["diversity_weight"]:
                    for ct in hparams["sparsity_weight"]:
                        _base = deepcopy(hparams)
                        _base["proximity_weight"] = pw
                        _base["diversity_weight"] = dw
                        _base["sparsity_weight"] = ct
                        hparams_to_save = _base.copy()
                        all_combinations.append((_base, hparams_to_save))
        else:
            raise ValueError("Unknown base counterfactual method")

        return all_combinations

    def _setup(self) -> None:
        self.cfg_gen = self.cfg["general"]
        self.cfg_exp = self.cfg["experiments_setup"]
        self.cfg_mod_hp = self.cfg["model_hyperparameters"]

        mm_dd = time.strftime("%m-%d_%H-%M-%S")
        salted_hash = str(abs(hash(str(self.cfg))) + int(time.time()))[:5]
        prefix = f"{mm_dd}_{salted_hash}"
        self.cfg_gen["result_path"] = os.path.join(self.cfg_gen["result_path"], prefix)
        self.cfg_gen["log_path"] = os.path.join(self.cfg_gen["log_path"], prefix)
        # self.cfg_gen["model_path"] = os.path.join(self.cfg_gen["model_path"])

        logging.debug(self.cfg)

        # Extract the results directory
        results_dir = self.cfg_gen["result_path"]
        results_df_dir = os.path.join(results_dir, "results")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(results_df_dir, exist_ok=True)

        # Save the config right away
        with open(os.path.join(results_dir, "config.yml"), "w") as file:
            yaml.dump(self.cfg, file)
