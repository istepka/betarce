import numpy as np 
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def create_two_donuts(n_samples: int = 1000, noise: float = 0.1, random_state: int = 0) -> np.ndarray:
    '''
    Create two donuts with the same number of samples and noise.
    
    Parameters:
        - n_samples: the number of samples for each donut (int)
        - noise: the noise of the donuts (float)
        - random_state: the random state (int)
    '''
    data = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    data2 = make_circles(n_samples=n_samples, noise=noise, random_state=random_state + 1, factor=0.5)
    
    X = np.concatenate([data[0], data2[0] / 1.5 + 1.6]) 
    y = np.concatenate([data[1], data2[1]])
    
    # Normalize the data to 0-1
    X = (X - X.min()) / (X.max() - X.min())
    
    return X, y


def plot_data(X: np.ndarray, y: np.ndarray):
    '''
    Plot the data.
    
    Parameters:
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
    '''
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()


if __name__ == '__main__':
    
    X, y = create_two_donuts()
    
    plot_data(X, y)