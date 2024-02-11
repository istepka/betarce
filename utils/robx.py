from copy import deepcopy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
from typing import Union

def counterfactual_stability(
    x: np.array, 
    pred_func: object, 
    variance: Union[np.ndarray, float] = 0.1,
    N: int = 100
    ) -> float:
    ''' 
    Calculate the Counterfactual stability metric for a given point x.
    
    Parameters:
        - x: the point for which the metric is calculated (np.array 1D)
        - pred_func: a function that returns the probability of belonging to class 1
        - variance: the variance of the data points in the training set (np.array 1D or float)
        - N: the number of samples to be generated (int)
    '''
    
    assert isinstance(x, np.ndarray), 'x must be a numpy array'
    assert isinstance(variance, (np.ndarray, float)), 'variance must be a numpy array or a float'
    if isinstance(variance, np.ndarray):
        assert len(variance) == len(x), 'variance must have the same length as x'
    
    cf_class = 1 if pred_func(x) > 0.5 else 0
    cov_mat = np.eye(len(x)) * variance
    X_p = np.random.multivariate_normal(x, cov_mat, size=(N,), check_valid='raise')
    
    X_pred = pred_func(X_p) if cf_class == 1 else 1 - pred_func(X_p) 
    
    c_mean = np.mean(X_pred)
    c_variance = np.mean((X_pred - c_mean) ** 2)  ** 0.5
    c_stability = c_mean - c_variance
    
    return c_stability

def counterfactual_stability_test(counterfactual_stability: float, tau: float) -> bool:
    '''
    Test if the counterfactual stability is higher than the threshold tau.
    
    Parameters:
        - counterfactual_stability: the counterfactual stability metric (float)
        - tau: the threshold (float)
    '''
    return counterfactual_stability > tau

def MLP_predict_class_1_proba(mlp, x) -> np.array:
    # Check if mlp is from sci-kit learn
    if isinstance(mlp, MLPClassifier):
        return mlp.predict_proba(x)[:, 1]
    
    # Check if mlp is from pytorch
    if isinstance(mlp, torch.nn.Module):
        return mlp(torch.Tensor(x)).detach().numpy()[:, 1]
    
def RF_predict_class_1_proba(rf: RandomForestClassifier, x) -> np.array:
    assert isinstance(rf, RandomForestClassifier), 'rf must be a RandomForestClassifier'
    return rf.predict_proba(x)[:, 1]

def get_conservative_counterfactuals(
    counterfactual: np.ndarray,
    data_X: np.ndarray,
    predict_class_proba_fn: object,
    variance: Union[np.ndarray, float] = 0.1,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    gamma: float = 0.5
    ) -> np.ndarray:
    '''
    Find k-nearest neighbors of the counterfactual that belong to the same class as the counterfactual. And that pass the counterfactual stability test.
    
    Parameters:
        - counterfactual: the counterfactual (np.array 1D)
        - data_X: the training data (np.array 2D)
        - predict_class_proba_fn: a function that returns the probability of belonging to class 1 (object)
        - variance: the variance of the data points in the training set (np.array 1D or float)
        - tau: the threshold (float)
        - N: the number of samples to be generated (int)
        - k: the number of neighbors (int)
        - gamma: the threshold for the class probability (float)
    '''
    cf_class = 1 if predict_class_proba_fn(counterfactual) > gamma else 0
    
    # Find k-nearest neighbors of the counterfactual that belong to the same class as the counterfactual
    data_Y = predict_class_proba_fn(data_X)
    data_Y = (data_Y > gamma).astype(int)
    correct_class_mask = data_Y == cf_class
    data = data_X[correct_class_mask]
    
    # Get the sorted indices of the neighbors by distance, from the closest to the farthest
    dist = np.sum(np.abs(data - counterfactual), axis=1)
    indices = np.argsort(dist)
    
    # Get the indices of the neighbors that belong to the same class as the counterfactual
    data = data[indices]
    
    conservative_counterfactuals = []
    history = []
    
    for x in data:
        cs = counterfactual_stability(x, predict_class_proba_fn, variance, N)
        history.append(cs)
        cst = counterfactual_stability_test(cs, tau)
        if cst:
            conservative_counterfactuals.append(x)
            
        if len(conservative_counterfactuals) == k:
            break
        
    if len(conservative_counterfactuals) < k:
        print('Warning: not enough neighbors pass the counterfactual stability test')
        print('Counterfactual stability history:', history)
        if len(conservative_counterfactuals) == 0:
            print('Warning: no neighbors pass the counterfactual stability test')
            return None
    
    return np.array(conservative_counterfactuals)
    
    

def robx_algorithm(
    X_train: np.ndarray,
    predict_class_proba_fn: object,
    start_counterfactual: np.ndarray,
    variance: Union[np.ndarray, float] = 0.1,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    robx_max_iter: int = 100,
    robx_lambda: float = 0.1,
    ) -> Union[np.ndarray, None]:
    '''
    The ROBX algorithm.
    
    Parameters:
        - X_train: the training data (np.array 2D)
        - predict_class_proba_fn: a function that returns the probability of belonging to class 1 (object)
        - start_counterfactual: the starting counterfactual (np.array 1D)
        - variance: the variance of the data points in the training set (np.array 1D or float)
        - tau: the threshold (float)
        - N: the number of samples to be generated (int)
        - k: the number of neighbors (int)
        - robx_max_iter: the maximum number of iterations (int)
        - robx_lambda: the lambda parameter (float) - it controls how much the counterfactual moves towards its 
            conservative counterfactual at each iteration. The closer to 1, the faster the counterfactual moves towards.
    '''
    
    # Initial check if the starting counterfactual passes the counterfactual stability test
    if counterfactual_stability_test(counterfactual_stability(start_counterfactual, predict_class_proba_fn, variance, N), tau):
        print('The starting counterfactual passes the counterfactual stability test')
        return start_counterfactual, None
    
    
    # Find k conservative counterfactuals
    conservative_counterfactuals = get_conservative_counterfactuals(
        counterfactual=start_counterfactual,
        data_X=X_train,
        predict_class_proba_fn=predict_class_proba_fn,
        variance=variance,
        tau=tau,
        N=N,
        k=k,
    )
    
    # If no conservative counterfactuals are found, return None
    # Typically this happens when either tau or/and variance (ie sampling radius) are too big
    if conservative_counterfactuals is None:
        return None, None
    
    # Create k copies of the starting counterfactual
    counterfactuals = [deepcopy(start_counterfactual) for _ in range(k)]
    
    # Optimization loop: 
    # - each counterfactual is optimized independently and moves towards its conservative counterfactual
    # - the optimization stops when the first counterfactual passes the counterfactual stability test
    # - pessimistically, the optimization stops when first counterfactual reaches its conservative counterfactual
    # - each step is a convex combination of the current counterfactual and its conservative counterfactual
    
    history = []
    for c in counterfactuals: history.append([c])
    
    for iteration in range(robx_max_iter):
        for i, (cf, cf_conservative) in enumerate(zip(counterfactuals, conservative_counterfactuals)):
            cf = robx_lambda * cf_conservative + (1 - robx_lambda) * cf # convex combination
            history[i].append(cf)
            counterfactuals[i] = cf
            
            if counterfactual_stability_test(counterfactual_stability(cf, predict_class_proba_fn, variance, N), tau):
                return cf, conservative_counterfactuals
             
    # If the optimization loop does not stop, return none
    return None, None
                 
