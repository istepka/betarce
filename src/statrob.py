from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from typing import Union
import pandas as pd
from joblib import Parallel, delayed
import scipy
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from create_data_examples import DatasetPreprocessor, Dataset
from explainers import growing_spheres_search, GrowingSpheresExplainer

# Scipy screams some radom stuff
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MLPClassifier(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list = [64, 64, 64],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 seed: int = 42
        ) -> None:
        '''
        input_dim: int, input dimension
        hidden_dims: list, hidden layer dimensions
        activation: str, activation function
        dropout: float, dropout rate
        '''
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        torch.manual_seed(seed)
        
        self.build_model()

    def build_model(self):
        input_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            else:
                raise ValueError('Invalid activation function')
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, self.output_dim))
        self.layers.append(nn.Sigmoid())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.flatten()
    
    def predict_proba(self, x):
        x = array_to_tensor(x)
        return self.forward(x).detach().numpy()
    
    def predict_crisp(self, x, threshold=0.5):
        x = array_to_tensor(x)
        pred = self.predict_proba(x)
        
        if isinstance(pred, np.ndarray):
            pred = array_to_tensor(pred)
            
        return (pred > threshold).int().detach().numpy()
    
    def fit(self, 
            X_train: Union[np.array, torch.Tensor],
            y_train: Union[np.array, torch.Tensor],
            X_val: Union[np.array, torch.Tensor] = None,
            y_val: Union[np.array, torch.Tensor] = None,
            epochs: int = 100,
            lr: float = 0.002, 
            batch_size: int = 256,
            verbose: bool = True,
            early_stopping: bool = True,
            device: str = 'cpu'
        ) -> None:
        '''
        '''
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Reshape y_train
        if len(y_train.shape) == 2:
            y_train = y_train.flatten()
            if y_val is not None:
                y_val = y_val.flatten()
                
        # If dataframes, convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
        
        X_train = array_to_tensor(X_train, device=device, dtype=torch.float32)
        y_train = array_to_tensor(y_train, device=device, dtype=torch.float32)
        if X_val is not None:
            X_val = array_to_tensor(X_val, device=device, dtype=torch.float32)
            y_val = array_to_tensor(y_val, device=device, dtype=torch.float32)
            
        val_loss_history = []
        early_stopping_patience = 5
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            
            if verbose and epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
            
            if early_stopping and X_val is not None:
                with torch.no_grad():
                    self.eval()
                    y_pred_val = self.forward(X_val)
                    val_loss = criterion(y_pred_val, y_val).item()
                    val_loss_history.append(val_loss)
                    
                    if verbose and epoch % 5 == 0:
                        print(f'Epoch: {epoch}, Validation Loss: {val_loss}')
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping due to validation loss not improving.")
                        break
                

                    
    def evaluate(self, 
                 X_test: Union[np.array, torch.Tensor],
                 y_test: Union[np.array, torch.Tensor],
                 device: str = 'cpu'
        ) -> float:
        '''
        X_test: np.array | torch.Tensor, test data
        y_test: np.array | torch.Tensor, test labels
        device: str, device to use
        '''
        self.eval()
        X_test = array_to_tensor(X_test, device=device)
        y_test = array_to_tensor(y_test, device=device)
        y_pred = self.predict_crisp(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        return accuracy, recall, precision, f1

def train_K_mlps(X_train, y_train, X_test, y_test, K: int = 5, evaluate: bool = True, bootstrap: bool = True):
    '''
    X_train: np.array, training data
    y_train: np.array, training labels
    X_test: np.array, test data
    y_test: np.array, test labels
    K: int, number of models to train
    bootstrap: bool, whether to use bootstrapping
    '''
    if bootstrap:
        X_train, y_train = bootstrap_data(X_train, y_train)
    
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    models = []
    for k in range(K):
        layers = np.random.randint(3, 5)
        dims = np.random.choice([64,128], size=layers)
        dropout = np.random.randint(1,5) / 10
        mlp = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=dims, activation='relu', dropout=dropout, seed=np.random.randint(1, 100) + k)
        mlp.fit(
            X_train, 
            y_train, 
            X_val=X_test, 
            y_val=y_test,
            verbose=False,
            early_stopping=True,
            lr=0.002,
            epochs=100
        )
        accuracy, recall, precision, f1 = mlp.evaluate(X_test, y_test)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        models.append(mlp)
    print(f'Average accuracy: {np.mean(accuracies)}, Average recall: {np.mean(recalls)}, Average precision: {np.mean(precisions)}, Average f1: {np.mean(f1s)}')
    return models, accuracies, recalls, precisions, f1s

def train_K_mlps_in_parallel(X_train, y_train, X_test, y_test, K: int = 20, n_jobs: int = 4, bootstrap: bool = True):
    '''
    X_train: np.array, training data
    y_train: np.array, training labels
    X_test: np.array, test data
    y_test: np.array, test labels
    K: int, number of models to train
    n_jobs: int, number of jobs to run in parallel
    bootstrap: bool, whether to use bootstrapping
    '''
    
    k_for_each_job = K // n_jobs 
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_K_mlps)(X_train, y_train, X_test, y_test, k_for_each_job, bootstrap=bootstrap) for _ in range(n_jobs)
    )
    return results

def ensemble_predict_proba(models: list[nn.Module], X: Union[np.ndarray, torch.Tensor]) -> list[float]:
    '''
    models: list, list of trained models
    X: np.array, data
    '''
    predictions = []
    X_tensor = array_to_tensor(X)
    for model in models:
        predictions.append(model.predict_proba(X_tensor))
    predictions = np.array(predictions)
    return predictions
                    
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
    
def estimate_beta_distribution(
    sample: np.ndarray,
    method: str = 'MLE',
    ) -> tuple[float, float]:
    '''
    Estimate beta distribution based on a sample
    
    Parameters:
        - sample (np.ndarray): a 1D array of size N
        - method (str): either 'MM' or 'MLE'. This decides the method of parameter estimation.
        
    Returns:
        - (alpha, beta) parameters of the estimated beta distribution 
    ''' 
    
    match method:
        case 'MM': 
            alpha, beta, _, _ = scipy.stats.beta.fit(sample, method='MM', floc=0, fscale=1)
        case 'MLE':
            alpha, beta, _, _ = scipy.stats.beta.fit(sample, method='MLE', floc=0, fscale=1)
        case _:
            raise ValueError(f'Estimation method not known: {method} should be either "MM" or "MLE"!')
    
    return alpha, beta        
  
def bootstrap_buckets(sample: np.ndarray, bootstrap_sample_size: int = 20, buckets: int = 30):
    return np.random.choice(sample, size=(buckets, bootstrap_sample_size), replace=True)

def bootstrap_data(X: np.ndarray, y: np.ndarray, bootstrap_sample_size_frac: int = 0.8) -> tuple[np.ndarray, np.ndarray]:
    '''
    Bootstrap the sample
    
    Parameters:
        - sample: np.ndarray, the sample to bootstrap
        - bootstrap_sample_size_frac: float, the fraction of the sample to use for bootstrapping
        
    Returns:
        - np.ndarray, the bootstrapped sample
    '''
    range_indices = np.arange(len(X))
    size = int(len(X) * bootstrap_sample_size_frac)
    indices = np.random.choice(range_indices, size=size, replace=False)
    return X[indices], y[indices]

def test_with_CI(sample: np.ndarray, confidence, thresh: float = 0.5, estimation_method: str = 'MLE') -> bool:
    alpha, beta = estimate_beta_distribution(sample, method=estimation_method)
    left, right = scipy.stats.beta.interval(confidence, alpha, beta)
    return left > thresh
  
def test_gs(sample: np.ndarray, pred_fn_crisp: callable, preprocessor: DatasetPreprocessor) -> np.ndarray:
    
    sample = sample.reshape(1, -1)
    
    _input = pd.DataFrame(sample, columns=preprocessor.X_train.columns)
    
    cf = growing_spheres_search(
        instance=_input,
        keys_mutable=preprocessor.X_train.columns.tolist(),
        keys_immutable=[],
        continuous_cols=preprocessor.continuous_columns,
        binary_cols=preprocessor.encoder.get_feature_names_out().tolist(),
        feature_order=preprocessor.X_train.columns.tolist(),
        pred_fn_crisp=pred_fn_crisp,
        n_search_samples=100,
    )
    
    return cf

def wrap_ensemble_crisp(sample: np.ndarray, models: list[nn.Module], method='avg-std') -> Union[int, np.ndarray]:
    '''
    Wrap the ensemble prediction function to be used in the test_gs function.  
    Specifically, reduce the ensemble predictions to a single value.
    
    Parameters:
        sample: np.ndarray, input example
        models: list[nn.Module], list of trained models
        method: str, method to use for reducing the ensemble predictions, either 'avg-std' or 'avg'
                the first one uses the average minus the standard deviation, the second one uses only the average
    
    Returns:
        float: reduced value
    '''
    pred_fn = lambda x: ensemble_predict_proba(models, x)
    match method:
        case 'avg-std':
            return ((pred_fn(sample).mean(axis=0) - pred_fn(sample).std(axis=0)) > 0.5).astype(int)
        case 'avg':
            return (pred_fn(sample).mean(axis=0) > 0.5).astype(int)
        case _:
            raise ValueError(f'Unknown method: {method}')

class StatrobGlobal:
    def __init__(self, dataset: Dataset, 
                 preprocessor: DatasetPreprocessor, 
                 blackbox: MLPClassifier,
                 seed: int = 42
        ) -> None:
 
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.models: list[nn.Module] = []
        self.blackbox = blackbox
        
    def fit(self, k_mlps: int = 32, _bootstrap: bool = False) -> None:
        '''
        Fit the ensemble of models
        
        Parameters:
            - k_mlps: int, number of models to train
            - _bootstrap: bool, whether to use bootstrapping
        '''
        X_train, X_test, y_train, y_test = self.preprocessor.get_numpy()
        
        print('Training the ensemble of models')
        results = train_K_mlps_in_parallel(X_train, y_train, X_test, y_test, K=k_mlps, n_jobs=1, bootstrap=_bootstrap)
        self.models = [model for model, _, _, _, _ in results]
        self.models = [model for sublist in self.models for model in sublist]
        print('Training the ensemble of models done')
        
    def __function_to_optimize(self, 
                               x: np.ndarray, 
                               target_class: int, 
                               beta_confidence: float, 
                               beta_estim_method: str ='MLE',
                               classification_threshold: float = 0.5
        ) -> Union[int, np.ndarray]:
        '''
        Return the value of the function at a given point x
        '''
        assert beta_confidence > 0 and beta_confidence < 1, 'Confidence level must be between 0 and 1'
        # Validity criterion
        blackbox_preds = self.blackbox.predict_crisp(x, classification_threshold)
        if isinstance(blackbox_preds, torch.Tensor):
            blackbox_preds = blackbox_preds.detach().numpy()
        blackbox_preds = blackbox_preds if target_class == 1 else 1 - blackbox_preds
        # print(f'Blackbox prediction: {blackbox_preds.shape}, {blackbox_preds.mean()} {blackbox_preds.flatten().round(3)} {self.blackbox.predict_proba(x).detach().numpy().flatten().round(3)}')
        # print('Blackbox preds shape', blackbox_preds.shape)
        
        # Skip if the blackbox predicts the wrong class
        if np.all(blackbox_preds == 0):
            if blackbox_preds.shape[0] > 1:
                return np.zeros(blackbox_preds.shape[0])
            else:
                return 0
        
        # Probabilistic outputs for beta CI test criterion
        preds = ensemble_predict_proba(self.models, x)
        preds = preds if target_class == 1 else 1 - preds
        # print('Preds shape', preds.shape) 
        
        test_mask = np.zeros(blackbox_preds.shape[0])
        
        # Estimate beta parameters
        if preds.shape[1] > 1:
            for i in range(preds.shape[1]):
                test_mask[i] = self.test_beta_credible_interval(
                    preds[:, i], 
                    confidence=beta_confidence, 
                    thresh=classification_threshold,
                    estimation_method=beta_estim_method
                )
        else:
            test_mask = self.test_beta_credible_interval(
                preds.flatten(),
                confidence=beta_confidence, 
                thresh=classification_threshold,
                estimation_method=beta_estim_method
            )
            
        results = test_mask.astype(bool) & blackbox_preds.astype(bool)
        return results.astype(int)
        
    def optimize(self, start_sample: np.ndarray, 
                 target_class: int, 
                 method: str = 'GS', 
                 desired_confidence: float = 0.9,
                 classification_threshold: float = 0.5,
                 estimation_method: str = 'MLE',
                 opt_hparams: dict = None
        ) -> tuple[np.ndarray, dict]:
        '''
        Optimize the input example
        
        Parameters:
            - start_sample: np.ndarray, the input example
            - target_class: int, the target class
            - method: str, the optimization method
            - desired_confidence: float, the desired confidence level
            - classification_threshold: float, the classification threshold
            - opt_hparams: dict, the optimization hyperparameters. If None, use defaults defined in the method
            
        Returns:
            - np.ndarray, the optimized input example
            - dict, some additional information if any
        '''
        
        artifact_dict = {
            'start_sample_passes_test': False,
            'counterfactual_does_not_pass_test': False,
            'counterfactual_does_not_have_target_class': False,
            'counterfactual_is_nan': False,
            'highest_confidence': np.nan,
        }
        
        pred_fn_crisp = lambda x: self.__function_to_optimize(x, 
            target_class=target_class, 
            beta_confidence=desired_confidence,
            classification_threshold=classification_threshold,
            beta_estim_method=estimation_method
        )
        
        # Check if the start sample is already a valid counterfactual
        if pred_fn_crisp(start_sample)[0] == 1:
            print('The start sample is already a valid statrob counterfactual')
            artifact_dict['start_sample_passes_test'] = True
            start_prds = ensemble_predict_proba(self.models, start_sample.reshape(1, -1))
            start_prds = start_prds if target_class == 1 else 1 - start_prds
            artifact_dict['highest_confidence'] = self.find_highest_confidence(
                start_prds,
                classification_threshold, 
                estimation_method
            )
            return start_sample, artifact_dict
            
        
        
        if method == 'GS':   
            # Use hparams if provided, otherwise use defaults
            if opt_hparams is None:
                opt_hparams = {}
            target_proba = opt_hparams['target_proba'] if 'target_proba' in opt_hparams else 0.5
            max_iter = opt_hparams['max_iter'] if 'max_iter' in opt_hparams else 100
            n_search_samples = opt_hparams['n_search_samples'] if 'n_search_samples' in opt_hparams else 1000
            p_norm = opt_hparams['p_norm'] if 'p_norm' in opt_hparams else 2
            step = opt_hparams['step'] if 'step' in opt_hparams else 0.1
            
            gs_explainer = GrowingSpheresExplainer(
                keys_mutable=self.preprocessor.X_train.columns.tolist(),
                keys_immutable=[],
                feature_order=self.preprocessor.X_train.columns.tolist(),
                binary_cols=self.preprocessor.encoder.get_feature_names_out().tolist(),
                continous_cols=self.preprocessor.continuous_columns,
                pred_fn_crisp=pred_fn_crisp,
                target_proba=target_proba,
                max_iter=max_iter,
                n_search_samples=n_search_samples,
                p_norm=p_norm,
                step=step
            )
            cf = gs_explainer.generate(start_sample)
        else:
            raise ValueError(f'Unknown method: {method}')  
        
        if cf is None or np.any(np.isnan(cf)):
            print(f'Counterfactual is not valid!: {cf}')
            artifact_dict['counterfactual_is_nan'] = True
            return None, artifact_dict
        
        # Posthoc check if the counterfactual is valid
        preds = ensemble_predict_proba(self.models, cf.reshape(1, -1))
        preds = preds if target_class == 1 else 1 - preds
        
        if not self.test_beta_credible_interval(preds.reshape(-1), confidence=desired_confidence, thresh=classification_threshold):
            print(f'Counterfactual does not pass the test!: \nCounterfactual {cf} \nPredictions: {preds.flatten()}')
            artifact_dict['counterfactual_does_not_pass_test'] = True
            
        pred = self.blackbox.predict_crisp(cf.reshape(1, -1), threshold=classification_threshold)
        if isinstance(pred, torch.Tensor):
            pred = int(pred.detach().numpy()[0])
        if isinstance(pred, np.ndarray):
            pred = int(pred[0])
        if pred != target_class:
            prob = self.blackbox.predict_proba(cf.reshape(1, -1))
            print(f'Counterfactual does not have the target class!: \nCounterfactual {cf} \nPrediction: {pred}, should be class: {target_class}. Proba: {prob}')
            artifact_dict['counterfactual_does_not_have_target_class'] = True
            
        artifact_dict['highest_confidence'] = self.find_highest_confidence(
                preds,
                classification_threshold, 
                estimation_method
            )
        
        return cf, artifact_dict
    
    def test_beta_credible_interval(self, 
            sample: np.ndarray, 
            confidence, thresh: float = 0.5, 
            estimation_method: str = 'MLE',
            epsilon: float = 1e-5
        ) -> bool:
        '''
        Test the beta distribution
        
        Parameters:
        - sample: np.ndarray, the array of predictions, has to be between (0,1) (exclusive)
        - confidence: float, the confidence level
        - thresh: float, the classification threshold
        - estimation_method: str, the estimation method for the beta distribution parameters
        - epsilon: float, the epsilon to clip the predictions. This is used to avoid zero or one values in the predictions,
                    which would make the beta distribution estimation impossible.
        '''
        # print('Sample to fit beta:', sample.tolist())
        _sample = np.clip(sample, epsilon, 1 - epsilon)
        
        result = test_with_CI(_sample, confidence, thresh, estimation_method)
        return result
    
    def find_highest_confidence(self, 
            preds: np.ndarray, 
            classification_threshold: float, 
            beta_estim_method: str = 'MLE',
            granularity: float = 0.01
        ) -> float:
        '''
        Find the highest confidence level at which the beta distribution test passes
        
        Parameters:
            - preds: np.ndarray, the array of predictions, has to be between (0,1) (exclusive)
            - classification_threshold: float, the classification threshold
            - beta_estim_method: str, the estimation method for the beta distribution parameters
            - granularity: float, the granularity of the search
        '''
        preds = preds.flatten()
        # print(f'Preds: {preds}')
        preds = np.clip(preds, 1e-5, 1 - 1e-5)
        alpha, beta = estimate_beta_distribution(preds, method=beta_estim_method)
        confidence = 0.99
        while confidence > 0:            
            left, _ = scipy.stats.beta.interval(confidence, alpha, beta)
            if left > classification_threshold:
                return confidence
            
            confidence -= granularity
            
        return 0
                
            
        
    
class StatRobXPlus:
    def __init__(self, dataset: Dataset, 
                 preprocessor: DatasetPreprocessor, 
                 blackbox: MLPClassifier,
                 seed: int = 42
        ) -> None:
 
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.models: list[nn.Module] = []
        self.blackbox = blackbox
        # self.nearest_neighbors = NearestNeighbors(n_neighbors=100, algorithm='auto', p=1)
        
    def fit(self, k_mlps: int = 32, 
            _bootstrap: bool = False,
            
        ) -> None:
        '''
        Fit the ensemble of models
        
        Parameters:
            - k_mlps: int, number of models to train
            - _bootstrap: bool, whether to use bootstrapping
        '''
        X_train, X_test, y_train, y_test = self.preprocessor.get_numpy()
        
        results = train_K_mlps_in_parallel(X_train, y_train, X_test, y_test, K=k_mlps, n_jobs=1, bootstrap=_bootstrap)
        self.models = [model for model, _, _, _, _ in results]
        self.models = [model for sublist in self.models for model in sublist]
        
        # self.nearest_neighbors = self.nearest_neighbors.fit(X_train)
        
        # Find k-nearest neighbors of the counterfactual that belong to the same class as the counterfactual
        self.data_Y = self.blackbox.predict_proba(self.preprocessor.X_train).flatten()
        if isinstance(self.data_Y, torch.Tensor):
            self.data_Y = self.data_Y.detach().numpy()
        
        
    def optimize(self, start_sample: np.ndarray, 
                 target_class: int, 
                 desired_confidence: float,
                 classification_threshold: float = 0.5,
                 lambd_update: float = 0.1,
                 max_iter: int = 100,
                 estimation_method: str = 'MLE',
                 opt_hparams: dict = None,
        ) -> Union[np.ndarray, None]:
         
        preds = ensemble_predict_proba(self.models, start_sample.reshape(1, -1))
        preds = preds if target_class == 1 else 1 - preds
        
        test = self.test_beta_credible_interval(preds, desired_confidence, classification_threshold)
        
        if test and self.blackbox.predict_crisp(start_sample) == target_class:
            print('The start sample passes the beta credible interval test')
            # makse sure that the repsone is onediemnsional
            return start_sample.flatten()
        
        # Find conservative counterfactuals
        conservative_cfs = self.find_conservative_counterfactuals(start_sample, target_class, desired_confidence, classification_threshold)
        
        candidates = np.repeat(start_sample, conservative_cfs.shape[0], axis=0)
        
        print(f'Conservative counterfactuals: {conservative_cfs.shape}')
        print(f'Candidates: {candidates.shape}')
        
        
        lambd = 0.1
        # Optimize the counterfactuals
        for iii in range(max_iter):
            for i, (cf, ccf) in enumerate(list(zip(candidates, conservative_cfs))):
                new_cf = (1-lambd) * cf + lambd * ccf
                
                # Check validity under the blackbox
                valid = self.blackbox.predict_crisp(new_cf)
                    
                validity = False
                if int(valid) == int(target_class):
                    validity = True
                
                preds = ensemble_predict_proba(self.models, new_cf.reshape(1, -1))
                preds = preds if target_class == 1 else 1 - preds
                
                test = self.test_beta_credible_interval(preds, desired_confidence, classification_threshold, estimation_method)
                if test and validity:
                    return new_cf
                else:
                    candidates[i] = new_cf
                    
                if np.allclose(new_cf, ccf):
                    return new_cf
            
            lambd = lambd + lambd_update
        
        return None
                
                
        
        
    def find_conservative_counterfactuals(self, 
                                          sample: np.ndarray, 
                                          target_class: int, 
                                          desired_confidence: float, 
                                          classification_threshold: float,
                                          k: int = 10
        ) -> np.ndarray:
    
        data_Y = (self.data_Y > classification_threshold).astype(int)
        # data_Y = data_Y.flatten()
        
        # Find the indices of the neighbors that belong to the same class as the counterfactual
        same_class_indices = np.where(data_Y == target_class)
        
        # Get the indices of the neighbors that belong to the same class as the counterfactual
        data = self.preprocessor.X_train.to_numpy()[same_class_indices]
        
        # Get the sorted indices of the neighbors by distance, from the closest to the farthest
        dist = np.sum(np.abs(data - sample), axis=1)
        indices = np.argsort(dist)
        
        # Get the indices of the neighbors that belong to the same class as the counterfactual
        data = data[indices]
        
        conservative_counterfactuals = []
        history = []
        
        for x in data:
            
            preds = ensemble_predict_proba(self.models, x.reshape(1, -1))
            preds = preds if target_class == 1 else 1 - preds
            
            test = self.test_beta_credible_interval(preds, desired_confidence, classification_threshold)
            
            if test:
                conservative_counterfactuals.append(x)

            if len(conservative_counterfactuals) == k:
                break
            
        if len(conservative_counterfactuals) < k:
            print('Warning: not enough neighbors pass the beta credible interval test')
            print('Counterfactual stability history:', history)
        if len(conservative_counterfactuals) == 0:
            print('Warning: no neighbors pass the beta credible interval test')
            return None
        
        return np.array(conservative_counterfactuals)
            
    def test_beta_credible_interval(self, sample: np.ndarray, confidence, thresh: float = 0.5, estimation_method: str = 'MLE') -> bool:
        '''
        Test the beta distribution
        '''
        result = test_with_CI(sample, confidence, thresh, estimation_method)
        return result
    
# UTILS
def plot_distribution_of_predictions(predictions: Union[list[float], np.ndarray], save_dir: Union[str, None] = None) -> None:
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if isinstance(predictions, list):
        preds = np.ndarray(predictions, dtype=np.float64)
    else:
        preds = predictions
        
    fig = plt.figure(figsize=(6,6))
    # plt.hist(preds, bins=50, density=True)
    sns.histplot(preds, kde=True)
    
    if save_dir:
        path = os.path.join(save_dir, 'distribution_of_predictions.png')
        plt.savefig(path, dpi=300)
        
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Distribution of predictions of different models for the same input example')
    
    plt.show()
    
def plot_grid_of_distribution_predictions(predictions: Union[list[list[float]], np.ndarray], 
                                          save_dir: Union[str, None] = None,
                                          rows: int = 3,
                                          cols: int = 5
    ) -> None:
    
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if isinstance(predictions, list):
            preds = np.ndarray(predictions, dtype=np.float16)
        else:
            preds = predictions.astype(np.float16)
            
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 8))
        axes = axes.flatten()
        
        for i in range(rows * cols):
            
            sns.histplot(preds[:, i], kde=True, ax=axes[i])
            
            # last row
            # if i // cols == rows - 1: 
            axes[i].set_xlabel('Probability')
                
            # if i % cols == 0:
            axes[i].set_ylabel('Count')
            
            axes[i].set_title(f'Example {i}')
            
        plt.suptitle('Distribution of predictions of different models for the same input example')
        plt.tight_layout()
        
        if save_dir:
            path = os.path.join(save_dir, 'grid_of_predictions_distributions.png')
            plt.savefig(path, dpi=300)
            
        plt.show()

def plot_beta(alpha: float,
    beta: float,
    sample_size: int = 10000,
    save_dir: Union[str, None] = None,
    ) -> None:
    
    samples = np.random.beta(alpha, beta, size=sample_size)
    
    
    sns.histplot(samples, kde=True)
    
    plt.suptitle(f'Beta distribution with a: {alpha:.2f}, b: {beta:.2f}')
    plt.xlabel('Beta value (probability)')
    plt.ylabel('Count')
    plt.tight_layout()
    
    plt.show()

def plot_beta_on_original(preds: np.ndarray, 
        alpha: float,
        beta: float,
        sample_size: int = 10000,
    ) -> None:
        
    fig, ax = plt.subplots(figsize=(6,6))
    # plt.hist(preds, bins=50, density=True)
    sns.histplot(preds, kde=True, ax=ax, color='orange', legend=True)
    
    plt.legend(['density', 'predictions'], loc='upper center')
    
    
    samples = np.random.beta(alpha, beta, size=sample_size)
    
    ax2 = ax.twinx()
    sns.histplot(samples, kde=True, ax=ax2, color='steelblue', legend=True)
    plt.legend(['beta distribution', 'beta samples'])
    
    
    plt.suptitle(f'Empirical predictions vs estimated beta distribution \n(with a: {alpha:.2f}, b: {beta:.2f})')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count of the empirical probability distribution')
    ax2.set_ylabel('Count of the estimated Beta distribution')
    plt.tight_layout()
    
    plt.show()
          
    
if __name__ == '__main__':
    if 'src' in os.getcwd():
        os.chdir('..')
        
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = Dataset('fico')

    preprocessor = DatasetPreprocessor(dataset, one_hot=True, random_state=SEED)

    X_train, X_test, y_train, y_test = preprocessor.get_numpy()
    
    blackbox = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=[64, 64, 64], activation='relu', dropout=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    blackbox.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=30, lr=0.001, batch_size=32, verbose=True, early_stopping=True)
    accuracy, recall, precision, f1 = blackbox.evaluate(X_test, y_test)
    print(f'BLACKBOX: Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}')
    
    # statrob = StatrobGlobal(dataset, preprocessor, blackbox=blackbox, seed=SEED)
    # statrob.fit(k_mlps=32)
    # start_sample = X_test[1:2]
    # target_class = blackbox.predict_crisp(array_to_tensor(start_sample)).item()
    # print(f'BLACKBOX Class: {target_class}')
    # cf = statrob.optimize(start_sample, target_class=1-target_class, method='GS')
    statrob = StatRobXPlus(dataset, preprocessor, blackbox=blackbox, seed=SEED)
    statrob.fit(k_mlps=32)
    start_sample = X_test[1:2]
    target_class = blackbox.predict_crisp(array_to_tensor(start_sample)).item()
    print(f'BLACKBOX Class: {target_class}')
    cf = statrob.optimize(start_sample, target_class=1-target_class)
    
    if cf is not None:
        cf = cf.reshape(1, -1)
        print(f'Counterfactual class: {blackbox.predict_crisp(array_to_tensor(cf)).item()}')
        print(f'Counterfactual: {cf}')
    else:
        print('No counterfactual found!')

    # print(f'Shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')
    
    # results = train_K_mlps_in_parallel(X_train, y_train, X_test, y_test, K=32, n_jobs=1)

    # accuracies = [accuracy for _, accuracy, _, _, _ in results]
    # accuracies = np.array(accuracies).flatten()
    # print(f'Accuracies: {accuracies}')
    # models = [model for model, _, _, _, _ in results]
    # models = [model for sublist in models for model in sublist]
    # print(f'Number of models: {len(models)}')
    
    # preds = ensemble_predict_proba(models, X_test)
    # print(preds.shape)
    # print(f'Ensemble Predictions: {preds[:, 7]}')
    
    # example_preds = preds[:, 8]
    # plot_distribution_of_predictions(example_preds, save_dir='images/statrob')
        
    # print(example_preds)

        
    # plot_grid_of_distribution_predictions(preds, save_dir='images/statrob')
    
    # s = preds[:, 4]
    # alpha, beta = estimate_beta_distribution(s, method='MLE')

    # alpha = np.clip(alpha, 0, 100)
    # beta = np.clip(beta, 0, 100)


    # print('Sample distribution')
    # plot_distribution_of_predictions(s)
    # print('Estimated Beta distribution')
    # plot_beta(alpha, beta)
    

    # plot_beta_on_original(s, alpha, beta)
    
    
    # s_bstrapped = bootstrap(s)
    # binary_test_results = []
    # for _s in s_bstrapped:
    #     res = test_with_CI(_s) 
    #     binary_test_results.append(res)
    # binary_test_results = np.array(binary_test_results, dtype=int)
    # # Binom
    # n = len(binary_test_results)
    # successes = binary_test_results.sum() 
    # p =  successes / n
    # print(f'p: {p:.2f}, n: {n}')

    # test_res = scipy.stats.binomtest(successes, n, p=0.5, alternative='greater')
    # print(test_res)
    
    
    # pred_function_crisp = lambda x: wrap_ensemble_crisp(x, models, method='avg-std')
    # test_gs(X_test[5], pred_function_crisp, preprocessor)