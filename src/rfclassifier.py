import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from utils import  bootstrap_data
from baseclassifier import BaseClassifier

class RFClassifier(BaseClassifier):
    
    model: RandomForestClassifier
    
    def __init__(self, hparams: dict) -> None:
        '''
        '''
        super(RFClassifier, self).__init__()
        
        self.hparams = hparams
        
        self.n_estimators = hparams['n_estimators']
        self.max_depth = hparams['max_depth']
        self.min_samples_split = hparams['min_samples_split']
        self.min_samples_leaf = hparams['min_samples_leaf']
        
        seed = hparams['seed']
        
        np.random.seed(seed)
        
        self.build_model()

    def build_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        
    def forward(self, x: np.ndarray) -> np.ndarray[float]:
        # if 1D, convert to 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.model.predict_proba(x)[:, 1].flatten()
    
    def predict_proba(self, x) -> np.ndarray[float]:
        assert isinstance(x, np.ndarray), 'Input must be a numpy array'
        return self.forward(x)
    
    def predict_crisp(self, x: np.ndarray, threshold=0.5) -> np.ndarray[int]:
        assert isinstance(x, np.ndarray), 'Input must be a numpy array'
        pred = self.predict_proba(x)
            
        return (pred > threshold).astype(int)
    
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
        
def train_random_forest(X_train, y_train, seed: int, hparams: dict, split: float = 0.8) -> tuple[RFClassifier, callable, callable]:
    '''
    Returns a trained model, a callable to predict probabilities, and a callable to predict crisp classes.
    '''
    
    logging.debug('Training RF model')
    
    # Initialize the model
    model = RFClassifier(hparams)
    
    # Create a validation set internally
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size=1 - split, random_state=seed)
    
    # Train the model
    model.fit(X_train1, y_train1)
    
    ret = model.evaluate(X_val1, y_val1)
    logging.debug(f'Validation set metrics: {ret}')
    
    predict_fn_1 = lambda x: model.predict_proba(x)
    predict_fn_1_crisp = lambda x: model.predict_crisp(x, threshold=hparams['classification_threshold'])
    
    return model, predict_fn_1, predict_fn_1_crisp

def train_K_RFs(X_train, 
        y_train, 
        X_test, 
        y_test, 
        hparams: dict, 
        fixed_hparams: bool,
        fixed_seed: int | None = None,
        bootstrap: bool = True,
        bootstrap_seed: int | None = None,
        K: int = 5, 
    ) -> dict:
    '''
    X_train: np.array, training data
    y_train: np.array, training labels
    X_test: np.array, test data
    y_test: np.array, test labels
    hparams: dict, hyperparameters
    K: int, number of models to train
    bootstrap: bool, whether to use bootstrapping
    bootstrap_seed: int | None, seed for bootstrapping
    '''
    assert bootstrap_seed is not None or not bootstrap, 'If bootstrapping, a seed must be provided'
    
    # Set up the lists to store the results
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    models = []
    
    # Train K models
    for k in range(K):
        
        model_hparams = hparams.copy()
            
        # Sample seed
        if fixed_seed is not None:
            seed = fixed_seed
        else:
            seed = np.random.randint(1, 100_000)
        
        
        # Bootstrap the data
        if bootstrap:
            np.random.seed(bootstrap_seed + k)    
            X_train, y_train = bootstrap_data(X_train, y_train)

        # Sample hyperparameters
        if not fixed_hparams:
            np.random.seed(bootstrap_seed + k) 
            model_hparams['n_estimators'] = np.random.choice(hparams['n_estimators'])
            model_hparams['max_depth'] = np.random.choice(hparams['max_depth'])
            model_hparams['min_samples_split'] = np.random.choice(hparams['min_samples_split'])
            model_hparams['min_samples_leaf'] = np.random.choice(hparams['min_samples_leaf'])
            
        model_hparams['seed'] = seed
        
        np.random.seed(seed)    
            
        rf = RFClassifier(model_hparams)
        
        # Train the model
        rf.fit(X_train, y_train)
        
        # Evaluate the model
        d = rf.evaluate(X_test, y_test)
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
        models.append(rf)
        
    print(f'Ensemble: Average accuracy: {np.mean(accuracies)}, Average recall: {np.mean(recalls)}, Average precision: {np.mean(precisions)}, Average f1: {np.mean(f1s)}')
    print(f'Ensemble: Std accuracy: {np.std(accuracies)}, Std recall: {np.std(recalls)}, Std precision: {np.std(precisions)}, Std f1: {np.std(f1s)}')
    return {
        'models': models,
        'accuracies': accuracies,
        'recalls': recalls,
        'precisions': precisions,
        'f1s': f1s
    }

def train_K_rfs_in_parallel(X_train, y_train, X_test, y_test, 
                            hparams: dict,
                            bootstrap: bool = True,
                            fixed_hparams: bool = False,
                            fixed_seed: int | None = None, 
                            K: int = 20, 
                            n_jobs: int = 4, 
    ) -> list[dict]:
    '''
    X_train: np.array, training data
    y_train: np.array, training labels
    X_test: np.array, test data
    y_test: np.array, test labels
    hparams: dict, hyperparameters
    fixed_hparams: bool, whether to use fixed hyperparameters
    fixed_seed: int | None, fixed seed
    K: int, number of models to train
    n_jobs: int, number of jobs
    '''
    
    k_for_each_job = K // n_jobs 
    print(f'Bootstrapping: {bootstrap}, Fixed hyperparameters: {fixed_hparams}, Fixed seed: {fixed_seed}, K: {K}, n_jobs: {n_jobs}')
    
    bootstrap_seed = 1234
    
    results = Parallel(n_jobs=n_jobs)(delayed(train_K_RFs)(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        hparams, 
        fixed_hparams,
        fixed_seed,
        bootstrap,
        bootstrap_seed,
        k_for_each_job
    ) for _ in range(n_jobs))
    
    return results
    

