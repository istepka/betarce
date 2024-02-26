# Adapted from https://github.com/carla-recourse/CARLA/blob/main/carla/recourse_methods/catalog/growing_spheres/library/gs_counterfactuals.py

import numpy as np
import pandas as pd
from numpy import linalg as LA
from .base_explainer import BaseExplainer

def hyper_sphere_coordindates(n_search_samples, instance, high, low, p_norm=2):

    # Implementation follows the Random Point Picking over a sphere
    # The algorithm's implementation follows: Pawelczyk, Broelemann & Kascneci (2020);
    # "Learning Counterfactual Explanations for Tabular Data" -- The Web Conference 2020 (WWW)
    # It ensures that points are sampled uniformly at random using insights from:
    # http://mathworld.wolfram.com/HyperspherePointPicking.html

    # This one implements the growing spheres method from
    # Thibaut Laugel et al (2018), "Comparison-based Inverse Classification for
    # Interpretability in Machine Learning" -- International Conference on Information Processing
    # and Management of Uncertainty in Knowledge-Based Systems (2018)

    """
    :param n_search_samples: int > 0
    :param instance: numpy input point array
    :param high: float>= 0, h>l; upper bound
    :param low: float>= 0, l<h; lower bound
    :param p: float>= 1; norm
    :return: candidate counterfactuals & distances
    """

    delta_instance = np.random.randn(n_search_samples, instance.shape[1])
    dist = np.random.rand(n_search_samples) * (high - low) + low  # length range [l, h)
    norm_p = LA.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta_instance = np.multiply(delta_instance, d_norm)
    candidate_counterfactuals = instance + delta_instance

    return candidate_counterfactuals, dist

def growing_spheres_search(
    instance,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    feature_order,
    pred_fn_crisp,
    n_search_samples=1000,
    p_norm=2,
    step=0.2,
    max_iter=1000,
):

    """
    :param instance: df
    :param step: float > 0; step_size for growing spheres
    :param n_search_samples: int > 0
    :param pred_fn_crisp
    :param p_norm: float=>1; denotes the norm (classical: 1 or 2)
    :param max_iter: int > 0; maximum # iterations
    :param keys_mutable: list; list of input names we can search over
    :param keys_immutable: list; list of input names that may not be searched over
    :return:
    """  #

    # correct order of names
    keys_correct = feature_order
    # divide up keys
    keys_mutable_continuous = list(set(keys_mutable) - set(binary_cols))
    keys_mutable_binary = list(set(keys_mutable) - set(continuous_cols))

    # Divide data in 'mutable' and 'non-mutable'
    # In particular, divide data in 'mutable & binary' and 'mutable and continuous'
    instance_immutable_replicated = np.repeat(
        instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_replicated = np.repeat(
        instance.values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_mutable_replicated_continuous = np.repeat(
        instance[keys_mutable_continuous].values.reshape(1, -1),
        n_search_samples,
        axis=0,
    )
    
    # init step size for growing the sphere
    low = 0
    high = low + step

    # counter
    count = 0
    counter_step = 1

    # get predicted label of instance
    instance_label = pred_fn_crisp(instance.values.reshape(1, -1))

    counterfactuals_found = False
    candidate_counterfactual_star = np.empty(
        instance_replicated.shape[1],
    )
    candidate_counterfactual_star[:] = np.nan
    while (not counterfactuals_found) and (count < max_iter):
        count = count + counter_step

        # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
        candidate_counterfactuals_continuous, _ = hyper_sphere_coordindates(
            n_search_samples, instance_mutable_replicated_continuous, high, low, p_norm
        )

        # sample random points from Bernoulli distribution
        candidate_counterfactuals_binary = np.random.binomial(
            n=1, p=0.5, size=n_search_samples * len(keys_mutable_binary)
        ).reshape(n_search_samples, -1)

        # make sure inputs are in correct order
        candidate_counterfactuals = pd.DataFrame(
            np.c_[
                instance_immutable_replicated,
                candidate_counterfactuals_continuous,
                candidate_counterfactuals_binary,
            ]
        )
        candidate_counterfactuals.columns = (
            keys_immutable + keys_mutable_continuous + keys_mutable_binary
        )
        # enforce correct order
        candidate_counterfactuals = candidate_counterfactuals[keys_correct]

        # STEP 2 -- COMPUTE l_1 DISTANCES
        if p_norm == 1:
            distances = np.abs(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        elif p_norm == 2:
            distances = np.square(
                (candidate_counterfactuals.values - instance_replicated)
            ).sum(axis=1)
        else:
            raise ValueError("Distance not defined yet")

        # counterfactual labels
        y_candidate = pred_fn_crisp(candidate_counterfactuals.values)
        indeces = np.where(y_candidate != instance_label)
        candidate_counterfactuals = candidate_counterfactuals.values[indeces]
        candidate_dist = distances[indeces]

        if len(candidate_dist) > 0:  # certain candidates generated
            min_index = np.argmin(candidate_dist)
            candidate_counterfactual_star = candidate_counterfactuals[min_index]
            counterfactuals_found = True

        # no candidate found & push search range outside
        low = high
        high = low + step

    return candidate_counterfactual_star


class GrowingSpheresExplainer(BaseExplainer):
    def __init__(self, keys_mutable: list, 
             keys_immutable: list, 
             feature_order: list, 
             binary_cols: list, 
             continous_cols: list,
             pred_fn_crisp: callable,
             target_proba: float, 
             target_class: int, 
             max_iter: int = 1000,
             n_search_samples: int = 1000, 
             p_norm: float = 2,
             step: float = 0.1
        ) -> None:
        '''
        Prepare the Growing Spheres explainer.
        
        Parameters:
            - keys_mutable: the mutable keys (list)
            - keys_immutable: the immutable keys (list)
            - feature_order: the feature order (list)
            - binary_cols: the binary columns (list)
            - continous_cols: the continuous columns (list)
            - pred_fn_crisp: the prediction function (callable)
            - target_proba: the target probability (float)
            - target_class: the target
            - max_iter: the maximum number of iterations (int)
            - n_search_samples: the number of search samples (int)
            - p_norm: the norm (float)
            - step: the step (float)
        '''
        
        self.keys_mutable = keys_mutable
        self.keys_immutable = keys_immutable
        self.feature_order = feature_order
        self.binary_cols = binary_cols
        self.continous_cols= continous_cols
        self.pred_fn = pred_fn_crisp
        self.target_proba = target_proba
        self.target_class = target_class
        self.max_iter = max_iter
        self.p_norm = p_norm
        self.step = step
        self.n_search_samples = n_search_samples

        self.prep_done = True
        
    def generate(self, query_instance: np.ndarray | pd.DataFrame):
        '''
        Generate the counterfactual.

        Parameters:
            - query_instance: the query instance (np.ndarray | pd.DataFrame)
        '''
        
        assert self.prep_done, 'You must prepare the explainer first'
        
        if isinstance(query_instance, np.ndarray):
            query_instance = pd.DataFrame(query_instance, columns=self.feature_order)
            
        return growing_spheres_search(
            instance=query_instance,
            keys_mutable=self.keys_mutable,
            keys_immutable=self.keys_immutable,
            continuous_cols=self.continous_cols,
            binary_cols=self.binary_cols,
            feature_order=self.feature_order,
            pred_fn_crisp=self.pred_fn,
            n_search_samples=self.n_search_samples,
            p_norm=self.p_norm,
            step=self.step,
            max_iter=self.max_iter,
        )
        