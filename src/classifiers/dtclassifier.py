import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Project imports
from .utils import bootstrap_data
from .baseclassifier import BaseClassifier

class DecisionTree(BaseClassifier):
    
    model: DecisionTreeClassifier
    
    def __init__(self, hparams: dict, seed: int) -> None:
        '''
        '''
        self.hparams = hparams
        
        self.max_depth = hparams['max_depth']
        self.min_samples_split = hparams['min_samples_split']
        self.min_samples_leaf = hparams['min_samples_leaf']
        self.criterion = hparams['criterion']
        self.max_features = hparams['max_features']
        self.random_state = seed

        np.random.seed(seed)
        
        self.build_model()

    def build_model(self):
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
    def forward(self, x: np.ndarray) -> np.ndarray[float]:
        # if 1D, convert to 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.model.predict_proba(x)[:, 1].flatten()
    
    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray[float]: 
        if isinstance(x, pd.DataFrame):
            x = x.values
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.forward(x)
    
    def predict_crisp(self, x: np.ndarray | pd.DataFrame, threshold=0.5) -> np.ndarray[int]:
        if isinstance(x, pd.DataFrame):
            x = x.values
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        pred = self.model.predict(x)
        return pred.astype(int).flatten()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Reshape y_train
        if len(y_train.shape) == 2:
            y_train = y_train.flatten()
                
        # If dataframes, convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self.model.fit(X_train, y_train)
                                
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        '''
        X_test: np.array | torch.Tensor, test data
        y_test: np.array | torch.Tensor, test labels
        device: str, device to use
        '''
        y_pred = self.predict_crisp(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        
def train_decision_tree(X_train, y_train, seed: int, hparams: dict, split: float = 0.8) -> tuple[DecisionTree, callable, callable]:
    '''
    Returns a trained model, a callable to predict probabilities, and a callable to predict crisp classes.
    '''
    
    logging.debug('Training Decision Tree model')
    
    # Initialize the model
    model = DecisionTree(hparams, seed)
    
    # Create a validation set internally
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size=1 - split, random_state=seed)
    
    # Train the model
    model.fit(X_train1, y_train1)
    
    ret = model.evaluate(X_val1, y_val1)
    logging.debug(f'Validation set metrics: {ret}')
    
    predict_fn_1 = lambda x: model.predict_proba(x)
    predict_fn_1_crisp = lambda x: model.predict_crisp(x, threshold=hparams['classification_threshold'])
    
    return model, predict_fn_1, predict_fn_1_crisp

def train_K_decision_trees(X_train, 
        y_train, 
        X_test, 
        y_test, 
        hparams_list: list[dict],
        seeds: list[int],
        bootstrap_seeds: list[int],
        K: int = 5, 
    ) -> dict:
    
    # Set up the lists to store the results
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    models = []
    
    # Train K models
    for k in range(K):
        
        seed = seeds[k]
        bootstrap_seed = bootstrap_seeds[k]
        model_hparams = hparams_list[k]
        
        np.random.seed(bootstrap_seed)
        X_train, y_train = bootstrap_data(X_train, y_train)
        
        
        print(f'Model {k+1} hyperparameters: {model_hparams}')
   
        dt = DecisionTree(model_hparams, seed)
        
        # Train the model
        dt.fit(X_train, y_train)
        
        # Evaluate the model
        d = dt.evaluate(X_test, y_test)
        accuracy = d['accuracy']
        recall = d['recall']
        precision = d['precision']
        f1 = d['f1']
        # logging.debug(f'Model {k+1} metrics: {accuracy}, {recall}, {precision}, {f1}')
        
        # Store the results
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        models.append(dt)
        
    print(f'Ensemble: Average accuracy: {np.mean(accuracies)}, Average recall: {np.mean(recalls)}, Average precision: {np.mean(precisions)}, Average f1: {np.mean(f1s)}')
    print(f'Ensemble: Std accuracy: {np.std(accuracies)}, Std recall: {np.std(recalls)}, Std precision: {np.std(precisions)}, Std f1: {np.std(f1s)}')
    return {
        'models': models,
        'accuracies': accuracies,
        'recalls': recalls,
        'precisions': precisions,
        'f1s': f1s
    }

def train_K_dts_in_parallel(X_train, 
                            y_train, 
                            X_test, 
                            y_test, 
                            hparamsB,
                            bootstrapB,
                            seedB, 
                            hparams_base: dict,
                            K: int,
                            n_jobs: int = 4, 
    ) -> list[dict]:
    
    k_for_each_job = K // n_jobs 
    
    hparamsB = [hparams_base | hp for hp in hparamsB]
    
    partitioned_hparams = np.array_split(hparamsB, n_jobs)
    partitioned_bootstrap = np.array_split(bootstrapB, n_jobs)
    partitioned_seed = np.array_split(seedB, n_jobs)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_K_decision_trees)(
            X_train, 
            y_train, 
            X_test, 
            y_test, 
            hparams_list, 
            seeds, 
            bootstrap_seeds, 
            k_for_each_job
        ) for hparams_list, seeds, bootstrap_seeds in zip(partitioned_hparams, partitioned_seed, partitioned_bootstrap)
    )
    
    return results
