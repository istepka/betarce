import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model(model: str,
                  X: np.ndarray, 
                  y: np.ndarray,
                  hparams: dict = None,
                  ) -> object:
    '''
    Train a model.
    
    Parameters:
        - model: the model (str) - 'MLP' or 'RF'
        - X: the data (np.ndarray)
        - y: the labels (np.ndarray)
        - hparams: the hyperparameters (dict)
    '''
    
    match model.upper():
        case 'MLP':
            model = MLPClassifier()
        case 'RF':
            model = RandomForestClassifier()
        case _:
            raise ValueError('model must be either MLP or RF')
        
    if hparams:
        model.set_params(**hparams)
    
    model.fit(X, y)
    return model

def train_calibrated_model(
            base_model: str,
            X: np.ndarray, 
            y: np.ndarray,
            base_model_hparams: dict = None,
            calibration_method: str = 'isotonic',
            random_state: int = 123,
        ) -> object:
        '''
        Train a calibrated model.
        
        Parameters:
            - base_model: the base model (str) - 'MLP' or 'RF'
            - X: the data (np.ndarray)
            - y: the labels (np.ndarray)
            - base_model_hparams: the hyperparameters for the base model (dict)
            - calibration_method: the calibration method (str) - 'isotonic' or 'sigmoid'
            - random_state: the random state (int)
        '''
        assert calibration_method in ['isotonic', 'sigmoid'], 'calibration_method must be either isotonic or sigmoid'
        assert base_model.upper() in ['MLP', 'RF'], 'base_model must be either MLP or RF'
        
        # X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3, random_state=random_state)
        
        match base_model.upper():
            case 'MLP':
                base_model = MLPClassifier(**base_model_hparams)
            case 'RF':
                base_model = RandomForestClassifier(**base_model_hparams)
            case _:
                raise ValueError('model must be either MLP or RF')
        
        # base_model.fit(X_train, y_train)
        
        model_isotonic = CalibratedClassifierCV(base_model, cv=3, method=calibration_method)
        model_isotonic.fit(X, y)
        
        return model_isotonic

def save_model(model: object, path: str) -> None:
    '''
    Save a model to a file.
    
    Parameters:
        - model: the model (object)
        - path: the path to save the model (str)
    '''
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, path)
    
def load_model(path: str) -> object:
    '''
    Load a model from a file.
    
    Parameters:
        - path: the path to load the model from (str)
    '''
    
    return joblib.load(path)

def scikit_predict_proba(model: object, x: np.ndarray) -> np.ndarray:
    '''
    Predict the probability of belonging to class 1.
    
    Parameters:
        - model: the model (object)
        - x: the data (np.ndarray)
    '''
    # print(x, x.shape)
    # Check dimensions
    if len(x.shape) == 1:
        _x = x.reshape(1, -1)
    else:
        _x = x
    # print(_x, _x.shape)
        
    if isinstance(model, MLPClassifier):
        return model.predict_proba(_x)[:, 1]
    
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(_x)[:, 1]
    
    # if model is from scikit learn still ok 
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(_x)[:, 1]
    
    raise ValueError('model must be either a MLPClassifier or a RandomForestClassifier')

def scikit_predict_proba_fn(model: object) -> object:
    '''
    Return a function that predicts the probability of belonging to class 1.
    
    Parameters:
        - model: the model (object)
    '''
    
    return lambda x: scikit_predict_proba(model, x)

def scikit_predict_crisp_fn(model: object) -> object:
    '''
    Return a function that predicts the class.
    
    Parameters:
        - model: the model (object)
    '''
    
    return lambda x: model.predict(x)


if __name__ == '__main__':
    from plot_helpers import plot_crisp_decision_boundary, plot_model_probabilites
    from create_data_examples import create_two_donuts
    
    X, y = create_two_donuts()
    
    # Train a MLP
    mlp_hparams = {
        'hidden_layer_sizes': (100, 100),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.01,
        'max_iter': 1000,
        'random_state': 0,
        'verbose': True,
        'early_stopping': True,
        }
    
    mlp = train_model('MLP', X, y, hparams=mlp_hparams)
    save_model(mlp, 'models/mlp.joblib')
    plot_crisp_decision_boundary(mlp, X, y, save_path='images/calibration/mlp_decision_boundary.png', show=False)
    plot_model_probabilites(mlp, X, y, save_path='images/calibration/mlp_probabilities.png', show=False)
    
    
    # Train a RF
    rf_hparams = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 0,
        }
    
    rf = train_model('RF', X, y, hparams=rf_hparams)
    save_model(rf, 'models/rf.joblib')
    plot_crisp_decision_boundary(rf, X, y, save_path='images/calibration/rf_decision_boundary.png', show=False)
    plot_model_probabilites(rf, X, y, save_path='images/calibration/rf_probabilities.png', show=False)

