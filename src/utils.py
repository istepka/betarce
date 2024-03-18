import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Union
                   
def array_to_tensor(X: Union[np.array, torch.Tensor, pd.DataFrame], 
                    device: str = 'cpu',
                    dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
    '''
    X: np.array, array to convert, or torch.Tensor
    device: str, device to use
    dtype: torch.dtype, data type
    '''
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=dtype)
    elif isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype=dtype)
        
    return X.to(device)

def bootstrap_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Bootstrap the sample, i.e., sample with replacement from the original sample 
    
    Parameters:
        - sample: np.ndarray, the sample to bootstrap
        
    Returns:
        - np.ndarray, the bootstrapped sample
    '''
    range_indices = np.arange(len(X))
    size = len(X)
    indices = np.random.choice(range_indices, size=size, replace=True)
    return X[indices], y[indices]

  