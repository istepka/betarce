import numpy as np 
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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

class DatasetPreprocessor:
    
    def __init__(self,
                name: str, 
                split: float = 0.8, 
                random_state: int = 0, 
                stratify: bool = False, 
                standardize_data: bool = False, 
                one_hot: bool = False) -> None:
        '''
        Initialize the dataset preprocessor.
        
        Parameters:
            - name: the name of the dataset (str) should be one of ['german', 'fico', 'compas']
            - split: the split of the dataset (float)
            - random_state: the random state (int)
            - stratify: whether to stratify the dataset (bool)
            - standardize_data: whether to standardize the dataset (bool)
            - one_hot: whether to one-hot encode the dataset (bool)
        '''
        
        self.name = name
        self.split = split
        self.random_state = random_state
        self.stratify = stratify
        self.standardize_data = standardize_data
        self.one_hot = one_hot
        
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        

        match self.name:
            case 'german':
                self.data = _german()
            case 'fico':
                self.data = _fico()
            case 'compas':
                self.data = _compas()
            case _:
                raise ValueError(f'Unknown dataset {self.name}')
            
        
        self.raw_df = self.data['raw_df']
        self.target_column = self.data['target_column']
        self.continuous_columns = self.data['continuous_columns']
        self.categorical_columns = self.data['categorical_columns']
        self.freeze_columns = self.data['freeze_columns']
        self.feature_ranges = self.data['feature_ranges']
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.__initial_transform_prep()

    def __initial_transform_prep(self) -> list[pd.DataFrame]:
        '''
        Prepare the initial transformation of the dataset.
        
        Returns:
            - the transformed dataset (list[pd.DataFrame])
        '''
        X = self.raw_df.drop(columns=[self.target_column])
        y = self.raw_df[self.target_column]
        
        # Drop rows that contain NaN values
        X = X.dropna(how='any', axis=0)
        y = y[X.index]
        
        if self.stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - self.split, random_state=self.random_state, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - self.split, random_state=self.random_state)
            
        # reset the index
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
            
        if self.standardize_data:
            X_train_s = self.standardize(X_train, self.continuous_columns, fit=True)
            X_test_s = self.standardize(X_test, self.continuous_columns, fit=False)
            
            X_train = X_train.drop(columns=self.continuous_columns)
            X_test = X_test.drop(columns=self.continuous_columns)
            
            X_train = pd.concat([X_train, X_train_s], axis=1)
            X_test = pd.concat([X_test, X_test_s], axis=1)
            
        if self.one_hot:
            X_train_o = self.one_hot_encode(X_train, self.categorical_columns, fit=True)
            X_test_o = self.one_hot_encode(X_test, self.categorical_columns, fit=False)
            
            X_train = X_train.drop(columns=self.categorical_columns)
            X_test = X_test.drop(columns=self.categorical_columns)
            
            X_train = pd.concat([X_train, X_train_o], axis=1)
            X_test = pd.concat([X_test, X_test_o], axis=1)
            
        return [X_train, X_test, y_train, y_test]
     
    def one_hot_encode(self, X: pd.DataFrame, categorical_columns: list[str], fit: bool) -> pd.DataFrame:
        '''
        One-hot encode the dataset.
        
        Parameters:
            - X: the dataset (pd.DataFrame)
            - categorical_columns: the categorical columns (list[str])
        '''
        if fit:
            self.encoder.fit(X[categorical_columns])
        
        X_transformed = self.encoder.transform(X[categorical_columns])
        X_transformed_features = self.encoder.get_feature_names_out(categorical_columns)
        X_transformed = pd.DataFrame(X_transformed, columns=X_transformed_features)
        
        return X_transformed
    
    def inverse_one_hot_encode(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Inverse one-hot encode the dataset.
        
        Parameters:
            - X: the dataset (pd.DataFrame)
        '''
        X_transformed = self.encoder.inverse_transform(X)
        return X_transformed
    
    def standardize(self, X: pd.DataFrame, continuous_columns: list[str], fit: bool) -> pd.DataFrame:
        '''
        Standardize the dataset.
        
        Parameters:
            - X: the dataset (pd.DataFrame)
            - continous_columns: the continous columns (list[str])
            - fit: whether to fit the scaler (bool)
        '''
        if fit:
            self.scaler.fit(X[continuous_columns])
        X_scaled = pd.DataFrame(self.scaler.transform(X[continuous_columns]), columns=continuous_columns)
        return X_scaled
    
    def inverse_standardize(self, X: pd.DataFrame, continuous_columns: list[str]) -> pd.DataFrame:
        '''
        Inverse standardize the dataset.
        
        Parameters:
            - X: the dataset (pd.DataFrame)
            - continous_columns: the continous columns (list[str])
        '''
        X_scaled = pd.DataFrame(
            self.scaler.inverse_transform(X[continuous_columns]), 
            columns=continuous_columns
        )
        return X_scaled


def _german(path: str = 'data/german.csv') -> dict:
    '''
    Load the german dataset.
    
    Parameters:
        - path: the path to the german dataset (str)    
    '''
    raw_df = pd.read_csv(path)
    categorical_columns = [
        'checking_status', 'credit_history', 'purpose', 
        'savings_status', 'employment', 'personal_status', 
        'other_parties', 'property_magnitude', 'other_payment_plans', 
        'housing', 'job', 'own_telephone', 'foreign_worker'
        ]
    continuous_columns = [
        'duration', 'credit_amount', 'installment_commitment', 
        'residence_since', 'age', 'existing_credits', 'num_dependents'
        ]
    target_column = 'class'

    monotonic_increase_columns = []
    monotonic_decrease_columns = []
    freeze_columns = ['foreign_worker']
    feature_ranges = {
        'duration': [int(raw_df['duration'].min()), int(raw_df['duration'].max())],
        'credit_amount': [int(raw_df['credit_amount'].min()), int(raw_df['credit_amount'].max())],
        'installment_commitment': [int(raw_df['installment_commitment'].min()), int(raw_df['installment_commitment'].max())],
        'residence_since': [int(raw_df['residence_since'].min()), int(raw_df['residence_since'].max())],
        'existing_credits': [int(raw_df['existing_credits'].min()), int(raw_df['existing_credits'].max())],
        'num_dependents': [int(raw_df['num_dependents'].min()), int(raw_df['num_dependents'].max())],
        'age': [18, int(raw_df['age'].max())],
    }
    
    data = {
        'raw_df': raw_df,
        'categorical_columns': categorical_columns,
        'continuous_columns': continuous_columns,
        'target_column': target_column,
        'monotonic_increase_columns': monotonic_increase_columns,
        'monotonic_decrease_columns': monotonic_decrease_columns,
        'freeze_columns': freeze_columns,
        'feature_ranges': feature_ranges
    }
    
    return data

def _fico(path: str = 'data/fico.csv') -> dict:
    '''
    Load the fico dataset.
    
    Parameters:
        - path: the path to the fico dataset (str)
    '''
    
    raw_df = pd.read_csv(path)
    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove('RiskPerformance')
    target_column = 'RiskPerformance'
    
    freeze_columns = ['ExternalRiskEstimate']
    feature_ranges = {
        'PercentTradesNeverDelq': [0, 100],
        'PercentInstallTrades': [0, 100],
        'PercentTradesWBalance': [0, 100],
    }
    monotonic_increase_columns = []
    monotonic_decrease_columns = []
    
    # In fico dataset negative values mean that the value is missing
    mask_negative = ~np.any(raw_df[continuous_columns] < 0, axis=1)
    raw_df = raw_df[mask_negative]
    
    data = {
        'raw_df': raw_df,
        'categorical_columns': categorical_columns,
        'continuous_columns': continuous_columns,
        'target_column': target_column,
        'monotonic_increase_columns': monotonic_increase_columns,
        'monotonic_decrease_columns': monotonic_decrease_columns,
        'freeze_columns': freeze_columns,
        'feature_ranges': feature_ranges
    }
    
    return data
    
def _compas(path: str = 'data/compas.csv') -> dict:
    '''
    Load the compas dataset.
    
    Parameters:
        - path: the path to the compas dataset (str)
    '''
    
    raw_df = pd.read_csv('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv')
    raw_df = raw_df[raw_df['type_of_assessment'] == 'Risk of Recidivism']

    features = [
        'sex', 'age', 'race', 'juv_fel_count', 
        'decile_score', 'juv_misd_count', 'juv_other_count', 
        'priors_count', 'c_days_from_compas', 'c_charge_degree',
        'two_year_recid'
        ]

    categorical_columns = ['sex', 'race', 'c_charge_degree']
    continuous_columns = ['age', 'juv_fel_count', 
                        'decile_score', 'juv_misd_count', 
                        'juv_other_count', 'priors_count', 
                        'c_days_from_compas']
    target_column = 'two_year_recid'
    freeze_columns = ['age', 'sex', 'race', 'c_charge_degree']

    feature_ranges = {
        'age': [18, 100],
        'decile_score': [0, 10],
    }

    raw_df = raw_df[features]
    raw_df = raw_df.dropna(how='any', axis=0)
    
    data = {
        'raw_df': raw_df,
        'categorical_columns': categorical_columns,
        'continuous_columns': continuous_columns,
        'target_column': target_column,
        'freeze_columns': freeze_columns,
        'feature_ranges': feature_ranges
    }
    
    return data


if __name__ == '__main__':
    
    # X, y = create_two_donuts()
    # plot_data(X, y)
    
    german_dataset = DatasetPreprocessor(name='german', standardize_data=True, one_hot=True)
    print(german_dataset.X_train.head())