import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
from typing import Union

def counterfactual_stability(
    x: np.array, 
    pred_func: object, 
    variance: Union[np.ndarray, float],
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
    predict_class_fn: object,
    variance: np.ndarray,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    ) -> np.ndarray:
    '''
    Find k-nearest neighbors of the counterfactual that belong to the same class as the counterfactual. And that pass the counterfactual stability test.
    
    Parameters:
        - counterfactual: the counterfactual (np.array 1D)
        - data_X: the training data (np.array 2D)
        - predict_class_fn: a function that returns the probability of belonging to class 1 (object)
        - k: the number of neighbors (int)
    '''
    cf_class = 1 if predict_class_fn(counterfactual) > 0.5 else 0
    
    # Find k-nearest neighbors of the counterfactual that belong to the same class as the counterfactual
    data_Y = predict_class_fn(data_X)
    
    # Get the indices of the k-nearest neighbors (L1 distance)
    dist = np.sum(np.abs(data_X - counterfactual), axis=1)
    indices = np.argsort(dist)[:k]
    
    # Get the indices of the neighbors that belong to the same class as the counterfactual
    indices = indices[data_Y[indices] == cf_class]
    
    # Get the neighbors that pass the counterfactual stability test
    X = data_X[indices]
    
    conservative_counterfactuals = []
    
    for x in X:
        if counterfactual_stability_test(counterfactual_stability(x, predict_class_fn, variance, cf_class, N), tau):
            conservative_counterfactuals.append(x)
            
        if len(conservative_counterfactuals) == k:
            break
        
    if len(conservative_counterfactuals) < k:
        print('Warning: not enough neighbors pass the counterfactual stability test')

        if len(conservative_counterfactuals) == 0:
            print('Warning: no neighbors pass the counterfactual stability test')
            return None
    
    return np.array(conservative_counterfactuals)
    