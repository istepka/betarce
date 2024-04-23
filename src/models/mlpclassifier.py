from typing import Union
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


from .utils import array_to_tensor, bootstrap_data

class MLPClassifier(nn.Module):
    def __init__(self, 
                input_dim: int, 
                hidden_layers: list,
                activation: str,
                neurons_per_layer: int,
                seed: int,
                dropout: float = 0.2
        ) -> None:
        '''
        input_dim: int, input dimension
        hidden_layers: list, hidden layer dimensions
        activation: str, activation function
        dropout: float, dropout rate
        '''
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = 1
        self.activation = activation
        self.dropout = dropout
        self.hidden_layers = [neurons_per_layer for _ in range(hidden_layers)]
        
        self.layers = nn.ModuleList()
        
        torch.manual_seed(seed)
        
        self.build_model()

    def build_model(self):
        input_dim = self.input_dim
        for hidden_dim in self.hidden_layers:
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
    
    def predict_proba(self, x) -> np.ndarray[float]:
        x = array_to_tensor(x)
        return self.forward(x).detach().numpy().flatten()
    
    def predict_crisp(self, x, threshold=0.5) -> np.ndarray[int]:
        x = array_to_tensor(x)
        pred = self.predict_proba(x)
        
        if isinstance(pred, np.ndarray):
            pred = array_to_tensor(pred)
            
        return (pred > threshold).int().detach().numpy().flatten()
    
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
            optimizer: str = 'adam',
            device: str = 'cpu'
        ) -> None:
        '''
        '''
        criterion = nn.BCELoss()
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError('Invalid optimizer')
        
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
                logging.debug(f'Epoch: {epoch}, Loss: {loss.item()}')
            
            if early_stopping and X_val is not None:
                with torch.no_grad():
                    self.eval()
                    y_pred_val = self.forward(X_val)
                    val_loss = criterion(y_pred_val, y_val).item()
                    val_loss_history.append(val_loss)
                    
                    if verbose and epoch % 5 == 0:
                        logging.debug(f'Epoch: {epoch}, Validation Loss: {val_loss}')
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logging.debug("Early stopping due to validation loss not improving.")
                        break
                                
    def evaluate(self, 
                 X_test: Union[np.array, torch.Tensor],
                 y_test: Union[np.array, torch.Tensor],
                 device: str = 'cpu'
        ) -> dict[str, float]:
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
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        
def train_neural_network(X_train, y_train, seed: int, hparams: dict, split: float = 0.8) -> tuple[MLPClassifier, callable, callable]:
    '''
    Returns a trained model, a callable to predict probabilities, and a callable to predict crisp classes.
    '''
    
    logging.debug('Training torch model')
    
    # Initialize the model
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_layers=hparams['hidden_layers'],
        activation=hparams['activation'],
        dropout=hparams['dropout'],
        neurons_per_layer=hparams['neurons_per_layer'],
        seed=seed
    )
    
    # Create a validation set internally
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size=1-split, random_state=seed)
    
    # Train the model
    model.fit(
        X_train1, y_train1,
        X_val=X_val1, y_val=y_val1,
        epochs=hparams['epochs'],
        lr=hparams['lr'],
        batch_size=hparams['batch_size'],
        verbose=hparams['verbose'],
        early_stopping=hparams['early_stopping'],
    )
    
    ret = model.evaluate(X_val1, y_val1)
    logging.debug(f'Validation set metrics: {ret}')
    
    
    predict_fn_1 = lambda x: model.predict_proba(x)
    predict_fn_1_crisp = lambda x: model.predict_crisp(x, threshold=hparams['classification_threshold'])
    
    
    return model, predict_fn_1, predict_fn_1_crisp

def train_K_mlps(X_train, 
        y_train, 
        X_test, 
        y_test, 
        hparams_list: list[dict],
        seeds: list[int],
        bootstrap_seeds: list[int],
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
  
    # Set up the lists to store the results
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    models = []
    
    # Fixed hyperparameters
    
    # Train K models
    for k in range(K):
        
        seed = seeds[k]
        bootstrap_seed = bootstrap_seeds[k]
        hparams = hparams_list[k]
            
        np.random.seed(bootstrap_seed)    
        X_train, y_train = bootstrap_data(X_train, y_train)

        hidden_layers = hparams['hidden_layers']
        activation = hparams['activation']
        neurons_per_layer = hparams['neurons_per_layer']
        optimizer = hparams['optimizer']
        dropout = hparams['dropout']
        epochs = hparams['epochs']
        lr = hparams['lr']
        batch_size = hparams['batch_size']
        verbose = hparams['verbose']
        early_stopping = hparams['early_stopping']
        class_threshold = hparams['classification_threshold']
            
        
        np.random.seed(seed)    
        torch.manual_seed(seed) 
            
        mlp = MLPClassifier(input_dim=X_train.shape[1], 
                            hidden_layers=hidden_layers,
                            neurons_per_layer=neurons_per_layer,
                            activation=activation, 
                            dropout=dropout, 
                            seed=seed
        )
        
        print(seed, bootstrap_seed, hparams)
        
        # Train the model
        mlp.fit(
            X_train, 
            y_train, 
            X_val=X_test, 
            y_val=y_test,
            verbose=verbose,
            early_stopping=early_stopping,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
        )
        
        # Evaluate the model
        d = mlp.evaluate(X_test, y_test)
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
        models.append(mlp)
        
    print(f'Ensemble: Average accuracy: {np.mean(accuracies)}, Average recall: {np.mean(recalls)}, Average precision: {np.mean(precisions)}, Average f1: {np.mean(f1s)}')
    print(f'Ensemble: Std accuracy: {np.std(accuracies)}, Std recall: {np.std(recalls)}, Std precision: {np.std(precisions)}, Std f1: {np.std(f1s)}')
    return {
        'models': models,
        'accuracies': accuracies,
        'recalls': recalls,
        'precisions': precisions,
        'f1s': f1s
    }

def train_K_mlps_in_parallel(X_train, 
                            y_train, 
                            X_test, 
                            y_test, 
                            hparamsB,
                            bootstrapB,
                            seedB, 
                            hparams_base: dict,
                            ex_type: str,
                            K: int,
                            n_jobs: int = 4, 
    ) -> list[dict]:
    
    k_for_each_job = K // n_jobs 
    # print(f'Bootstrapping: {bootstrap}, Fixed hyperparameters: {fixed_hparams}, Fixed seed: {fixed_seed}, K: {K}, n_jobs: {n_jobs}')
    
    # append the base hyperparameters
    hparamsB = [hparams_base | h for h in hparamsB]
    
    partitioned_hparams = np.array_split(hparamsB, n_jobs)
    partitioned_bootstrap = np.array_split(bootstrapB, n_jobs)
    partitioned_seed = np.array_split(seedB, n_jobs)
 
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_K_mlps)(X_train, y_train, X_test, y_test,
                                hparams_list=hparams,
                                seeds=seeds,
                                bootstrap_seeds=bootstrap_seeds,
                                K=k_for_each_job
            ) for hparams, seeds, bootstrap_seeds in zip(partitioned_hparams, partitioned_seed, partitioned_bootstrap)
    )
    return results

def ensemble_predict_proba(models: list[MLPClassifier], X: Union[np.ndarray, torch.Tensor]) -> np.ndarray[float]:
    '''
    models: list, list of trained models
    X: np.array, data. Shape (1, n_features).
    
    Returns:
    np.ndarray, the predicted probabilities of shape (n_models,)
    '''
    assert len(models) > 0, 'No models to predict'
    if len(X.shape) == 2:
        assert X.shape[0] == 1, 'X should have shape (1, n_features)'
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    predictions = []
    X_tensor = array_to_tensor(X)
    for model in models:
        predictions.append(model.predict_proba(X_tensor))
    predictions = np.array(predictions)
    
    # Flatten the array if necessary
    if len(predictions.shape) == 2:
        predictions = predictions.flatten()
    
    return predictions
  
def ensemble_predict_crisp(sample: np.ndarray, models: list[MLPClassifier], class_threshold: float = 0.5) -> np.ndarray[int]:
    '''
    Wrapper function to predict crisp classes using an ensemble of models.

    Parameters:
        - sample: np.ndarray, the sample to predict
        - models: list, list of trained models
        - class_threshold: float, threshold for crisp classes
        
    Returns:
        - np.ndarray, the crisp class predictions of shape (n_models,)
    '''
    assert len(models) > 0, 'No models to predict'
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    if len(sample.shape) == 2:
        assert sample.shape[0] == 1, 'X should have shape (1, n_features)'

    predictions = []
    for model in models:
        predictions.append(model.predict_crisp(sample, threshold=class_threshold))
    predictions = np.array(predictions)
    
    if len(predictions.shape) == 2:
        predictions = predictions.flatten()
    
    return predictions

