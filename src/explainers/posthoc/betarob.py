import numpy as np
import pandas as pd
import logging
from scipy import stats

from .posthoc_explainer import PostHocExplainer
from ..base import GrowingSpheresExplainer
from ...datasets import Dataset, DatasetPreprocessor


def prepare_growing_spheres(
    preprocessor: DatasetPreprocessor, optimized_fn_crisp: callable, gs_hparams: dict
) -> GrowingSpheresExplainer:
    target_proba = gs_hparams["target_proba"]
    max_iter = gs_hparams["max_iter"]
    n_search_samples = gs_hparams["n_search_samples"]
    p_norm = gs_hparams["p_norm"]
    step = gs_hparams["step"]

    gsExplainer = GrowingSpheresExplainer(
        keys_mutable=preprocessor.X_train.columns.tolist(),
        keys_immutable=[],
        feature_order=preprocessor.X_train.columns.tolist(),
        binary_cols=preprocessor.encoder.get_feature_names_out().tolist(),
        continous_cols=preprocessor.continuous_columns,
        pred_fn_crisp=optimized_fn_crisp,
        target_proba=target_proba,
        max_iter=max_iter,
        n_search_samples=n_search_samples,
        p_norm=p_norm,
        step=step,
    )

    return gsExplainer


def prep_done(func):
    def wrapper(self, *args, **kwargs):
        assert self.is_prep_done, "You must prepare the explainer first"
        return func(self, *args, **kwargs)

    return wrapper


class BetaRob(PostHocExplainer):
    # attributes
    dataset: Dataset
    preprocessor: DatasetPreprocessor
    estimators_crisp: list[callable]
    seed: int
    growing_spheres_hparams: dict
    growingSpheresOptimizer: GrowingSpheresExplainer
    pred_fn_crisp: callable
    pred_fn_proba: callable
    classification_threshold: float

    # flags
    is_prep_done: bool = False

    def __init__(
        self,
        dataset: Dataset,
        preprocessor: DatasetPreprocessor,
        pred_fn_crisp: callable,
        pred_fn_proba: callable,
        estimators_crisp: list[callable],
        grow_sphere_hparams: dict,
        classification_threshold: float,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.estimators_crisp = estimators_crisp
        self.growing_spheres_hparams = grow_sphere_hparams
        self.pred_fn_crisp = pred_fn_crisp
        self.pred_fn_proba = pred_fn_proba
        self.classification_threshold = classification_threshold

    def prep(self) -> None:
        """
        Prepare the explainer.
        """
        self.is_prep_done = True

    @prep_done
    def generate(
        self,
        start_instance: np.ndarray | pd.DataFrame,
        target_class: int,
        delta: float,
        beta: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate the counterfactual.
        """
        self.target_class = target_class

        # Prepare the artifact dictionary to store the results for analytics
        artifact_dict = {
            "start_sample_passes_test": 0,
            "counterfactual_does_not_pass_test": 0,
            "counterfactual_does_not_have_target_class": 0,
            "counterfactual_is_nan": 0,
            "highest_delta": np.nan,
            "lower_bound_beta": np.nan,
            "upper_bound_beta": np.nan,
        }

        # Prepare the function to optimize
        optimized_fn_crisp = lambda instance: self.__function_to_optimize(
            instance=instance,
            delta_target=delta,
            beta_confidence=beta,
        )

        # Prepare the Growing Spheres optimizer
        self.growingSpheresOptimizer = prepare_growing_spheres(
            preprocessor=self.preprocessor,
            optimized_fn_crisp=optimized_fn_crisp,
            gs_hparams=self.growing_spheres_hparams,
        )

        # Check if the start instance passes the goal function
        if optimized_fn_crisp(start_instance) == 1:
            artifact_dict["start_sample_passes_test"] = 1
            artifact_dict["highest_delta"] = self.__get_left_bound_beta(
                start_instance, beta
            )
            artifact_dict["lower_bound_beta"], artifact_dict["upper_bound_beta"] = (
                self.__get_credible_interval_bounds(start_instance, beta)
            )
            return start_instance, artifact_dict

        # Generate the robust counterfactual
        robust_counterfactual = self.growingSpheresOptimizer.generate(
            query_instance=start_instance
        )

        # Check if counterfactual is found properly
        if robust_counterfactual is None or np.any(np.isnan(robust_counterfactual)):
            logging.warning(f"Counterfactual is NaN!: {robust_counterfactual}")
            artifact_dict["counterfactual_is_nan"] = 1
            return None, artifact_dict

        # Check if the counterfactual passes the goal function
        if optimized_fn_crisp(robust_counterfactual) != 1:
            logging.warning(
                f"Counterfactual does not pass the goal function!: {robust_counterfactual}"
            )
            artifact_dict["counterfactual_does_not_pass_test"] = 1
            return None, artifact_dict

        # Check if the counterfactual has the target class
        if self.__get_blackbox_pred_crisp(robust_counterfactual)[0] != 1:
            logging.warning(
                f"Counterfactual does not have the target class!: {robust_counterfactual}"
            )
            artifact_dict["counterfactual_does_not_have_target_class"] = 1
            return None, artifact_dict

        artifact_dict["highest_delta"] = self.__get_left_bound_beta(
            robust_counterfactual, beta
        )
        artifact_dict["lower_bound_beta"], artifact_dict["upper_bound_beta"] = (
            self.__get_credible_interval_bounds(robust_counterfactual, beta)
        )

        return robust_counterfactual, artifact_dict

    def __function_to_optimize(
        self, instance: np.ndarray, delta_target: float, beta_confidence: float
    ) -> np.ndarray[int] | int:
        """
        The function to be satisfied by the counterfactual.
        Reuturns 1 if the instance satisfies the function, 0 otherwise.
        """
        assert (
            beta_confidence > 0 and beta_confidence < 1
        ), "Beta confidence must be in (0, 1)"
        # assert len(instance.shape) == 2, 'Instance must be 2D'
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Constraint #1: Validity in the original model

        # First check if the instance has the target class
        model_preds: np.ndarray = self.__get_blackbox_pred_crisp(instance)
        if np.all(model_preds != 1):
            if instance.shape[0] == 1:
                return 0
            else:
                return np.zeros(instance.shape[0], dtype=int)
        model_preds = model_preds.astype(bool).flatten()

        # Constraint #2: Delta-robustness
        delta_results = np.zeros(instance.shape[0]).astype(bool)
        lbs = np.zeros(instance.shape[0])

        for i in range(instance.shape[0]):
            delta_results[i] = self.__test_delta_robustness(
                instance=instance[i],
                beta_confidence=beta_confidence,
                delta_target=delta_target,
            )
            left_bound = self.__get_left_bound_beta(instance[i], beta_confidence)
            lbs[i] = left_bound

        # AND the constraints
        final_results: np.ndarray = model_preds & delta_results

        # Return final results as 0 or 1
        if instance.shape[0] == 1:
            return final_results.astype(int)[0]
        else:
            return final_results.astype(int)

    def __test_delta_robustness(
        self,
        instance: np.ndarray,
        beta_confidence: float,
        delta_target: float,
    ) -> bool:
        """
        Test the delta-robustness.
        """
        left_bound = self.__get_left_bound_beta(instance, beta_confidence)
        # Check if the lower bound is greater than the delta_target
        return left_bound > delta_target

    def __get_left_bound_beta(
        self,
        instance: np.ndarray,
        beta_confidence: float,
    ) -> float:
        """
        Get the left bound.
        """
        # Initialize the alpha and beta prior parameters
        alpha, beta = 0.5, 0.5

        # Get the predictions of estimators (empirical samples)
        preds = self.__get_estimators_predictions(instance)
        ones = np.sum(preds == 1)
        zeros = np.sum(preds == 0)

        # Update the alpha and beta posterior parameters
        alpha += ones
        beta += zeros

        # Get the lower bound of credible interval for p from Beta
        left_bound = stats.beta.ppf(1 - beta_confidence, alpha, beta)
        return left_bound

    def __get_credible_interval_bounds(
        self,
        instance: np.ndarray,
        beta_confidence: float,
    ) -> tuple:
        """
        Get the credible interval bounds.
        """
        # Initialize the alpha and beta prior parameters
        alpha, beta = 0.5, 0.5

        # Get the predictions of estimators (empirical samples)
        preds = self.__get_estimators_predictions(instance)
        ones = np.sum(preds == 1)
        zeros = np.sum(preds == 0)

        # Update the alpha and beta posterior parameters
        alpha += ones
        beta += zeros

        # Get the credible interval for p from Beta
        lb, rb = stats.beta.interval(beta_confidence, alpha, beta)
        return lb, rb

    def __get_estimators_predictions(self, instance: np.ndarray) -> np.ndarray:
        """
        Get the predictions of the estimators.
        """
        preds = []
        for estimator in self.estimators_crisp:
            preds.append(estimator(instance))
        preds_all = np.array(preds, dtype=int).flatten()

        if self.target_class == 0:
            preds_all = 1 - preds_all

        return preds_all

    def __get_blackbox_pred_crisp(self, instance: np.ndarray) -> np.ndarray[int]:
        """
        Get the crisp prediction of the blackbox model.
        """
        if self.target_class == 0:
            return 1 - self.pred_fn_crisp(instance)
        else:
            return self.pred_fn_crisp(instance)

    def __get_blackbox_pred_proba(self, instance: np.ndarray) -> np.ndarray[float]:
        """
        Get the probabilistic prediction of the blackbox model.
        """
        if self.target_class == 0:
            return 1 - self.pred_fn_proba(instance)
        else:
            return self.pred_fn_proba(instance)
