import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import torch
import pandas as pd

from sklearn.datasets import make_circles


def create_two_donuts(
    n_samples: int = 1000, noise: float = 0.1, random_state: int = 0
) -> np.ndarray:
    """
    Create two donuts with the same number of samples and noise.

    Parameters:
        - n_samples: the number of samples for each donut (int)
        - noise: the noise of the donuts (float)
        - random_state: the random state (int)
    """
    data = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    data2 = make_circles(
        n_samples=n_samples, noise=noise, random_state=random_state + 1, factor=0.5
    )

    X = np.concatenate([data[0], data2[0] / 1.5 + 1.6])
    y = np.concatenate([data[1], data2[1]])

    # Normalize the data to 0-1
    X = (X - X.min()) / (X.max() - X.min())

    return X, y


def plot_data(X: np.ndarray, y: np.ndarray):
    """
    Plot the data.

    Parameters:
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def array_to_tensor(
    X: Union[np.array, torch.Tensor, pd.DataFrame],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    X: np.array, array to convert, or torch.Tensor
    device: str, device to use
    dtype: torch.dtype, data type
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=dtype)
    elif isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype=dtype)

    return X.to(device)


def bootstrap_data(
    X: np.ndarray, y: np.ndarray, seed=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap the sample, i.e., sample with replacement from the original sample

    Parameters:
        - sample: np.ndarray, the sample to bootstrap

    Returns:
        - np.ndarray, the bootstrapped sample
    """

    if seed is not None:
        np.random.seed(seed)

    range_indices = np.arange(len(X))
    size = len(X)
    indices = np.random.choice(range_indices, size=size, replace=True)
    return X[indices], y[indices]
