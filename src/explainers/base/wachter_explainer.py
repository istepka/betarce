import numpy as np
import pandas as pd
from alibi.explainers.counterfactual import Counterfactual
import sklearn

from .base_explainer import BaseExplainer


class AlibiWachter(BaseExplainer):
    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        dataset: pd.DataFrame,
        outcome_name: str,
        continuous_features: list = None,
    ) -> None:
        """
        Initialize the BaseExplainer.

        Parameters:
            - model: the model (object)
            - outcome_name: the outcome name (str)
            - continuous_features: the continuous features (list) - if None, all features are considered continuous
        """

        self.model = model
        self.dataset = dataset
        self.outcome_name = outcome_name
        self.continuous_features = continuous_features

        self.explainer: Counterfactual = None
        self.prep_done = False

    def prep(
        self,
        query_instance_shape: tuple,
        pred_fn: callable,
        feature_ranges: tuple = (0, 1),
        eps: object | float = 0.1,
        target_proba: float = 0.5,
        target_class: int | str = "other",
        max_iter: int = 1000,
        early_stop: int = 200,
        lam_init: float = 0.1,
        max_lam_steps: int = 10,
        tolerance: float = 0.01,
        learning_rate_init: float = 0.1,
        distance_fn: str = "l1",
    ) -> None:
        """
        Prepare the Wachter explainer.

        Parameters:
            - query_instance_shape: shape of single instance e.g. (1, 85)
            - eps: very important parameter - how each variable can be changed at one step. should be np.array of these values e.g. [0.1, 0.1, 1.0, 1.0].
            - feature_ranges: the feature ranges (tuple) - e.g. ((0, 1), (0, 1), (0, 1), (0, 1))
            - pred_fn: the prediction function (callable)
            - target_proba: the target probability (float)
            - target_class: the target
            - max_iter: the maximum number of iterations (int)
            - early_stop: the early stopping (int)
            - lam_init: the initial lambda (float)
            - max_lam_steps: the maximum lambda steps (int)
            - tolerance: the tolerance (float)
            - learning_rate_init: the initial learning rate (float)
            - distance_fn: the distance function (str) - 'l1' or 'l2'
        """

        if issubclass(self.model.__class__, sklearn.base.BaseEstimator):
            self.explainer = Counterfactual(
                pred_fn,
                query_instance_shape,
                distance_fn=distance_fn,
                target_proba=target_proba,
                target_class=target_class,
                max_iter=max_iter,
                early_stop=early_stop,
                lam_init=lam_init,
                max_lam_steps=max_lam_steps,
                tol=tolerance,
                learning_rate_init=learning_rate_init,
                feature_range=feature_ranges,
                init="identity",
                decay=True,
                write_dir=None,
                debug=False,
            )
        else:
            self.explainer = Counterfactual(
                self.model,
                query_instance_shape,
                distance_fn=distance_fn,
                target_proba=target_proba,
                target_class=target_class,
                max_iter=1000,
                early_stop=200,
                lam_init=0.1,
                max_lam_steps=10,
                tol=tolerance,
                learning_rate_init=0.1,
                feature_range=feature_ranges,
                init="identity",
                decay=True,
                write_dir=None,
                debug=False,
            )

        self.prep_done = True

    def generate(
        self,
        query_instance: pd.DataFrame | np.ndarray,
        total_cfs: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate counterfactuals for query instance

        `total_cfs` - number of counterfactuals to generate.
        If more than 1 then random sample is taken from all found
        counterfactuals in the optimization process
        """
        if not self.prep_done:
            raise ValueError("You need to call prep method first")

        if isinstance(query_instance, pd.DataFrame):
            explanation = self.explainer.explain(query_instance.to_numpy())
        else:
            explanation = self.explainer.explain(query_instance)

        wachter_counterfactuals = []

        if total_cfs > 1:
            # Get counterfactuals from the optimization process
            for _, lst in self.explanation["data"]["all"].items():
                if lst:
                    for cf in lst:
                        wachter_counterfactuals.append(cf["X"])

            # If no counterfactuals found return none
            if len(wachter_counterfactuals) == 0:
                return None

            # Reshape to (n, features)
            wachter_counterfactuals = np.array(wachter_counterfactuals).reshape(
                -1, query_instance.shape[1]
            )

            # Get random sample from all cfs to get desired number
            _indices_to_take = np.random.permutation(wachter_counterfactuals.shape[0])[
                0 : total_cfs - 1
            ]
            wachter_counterfactuals = wachter_counterfactuals[_indices_to_take, :]

            # Concat sample with the one counterfactual that wachter chose as best found
            wachter_counterfactuals = np.concatenate(
                [wachter_counterfactuals, explanation.cf["X"]], axis=0
            )
        else:
            wachter_counterfactuals = explanation.cf["X"]

        return pd.DataFrame(wachter_counterfactuals, columns=query_instance.columns)
