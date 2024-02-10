import numpy as np 
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder


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
    
class Dataset:
        def __init__(self, name: str, random_state: int = 0) -> None:
            '''
            Initialize the dataset.
            
            Parameters:
                - name: the name of the dataset (str) should be one of ['german', 'fico', 'compas', 'donuts', 'moons']
                - random_state: the random state (int)
            '''
            
            self.name = name
            self.random_state = random_state
            
            match self.name:
                case 'german':
                    self.data = _german()
                case 'fico':
                    self.data = _fico()
                case 'compas':
                    self.data = _compas()
                case 'donuts':
                    self.data = _donuts()
                case 'moons':
                    self.data = _moons()
                case _:
                    raise ValueError(f'Unknown dataset {self.name}')
                
            self.raw_df = self.data['raw_df']
            self.target_column = self.data['target_column']
            self.continuous_columns = self.data['continuous_columns']
            self.categorical_columns = self.data['categorical_columns']
            self.freeze_columns = self.data['freeze_columns']
            self.feature_ranges = self.data['feature_ranges']
            
        def __str__(self) -> str:
            return f'Dataset_{self.name}'
        
        def __repr__(self) -> str:
            return f'Dataset_{self.name}'
        
        def __len__(self) -> int:
            return len(self.raw_df)
        
        def __getitem__(self, index: int) -> pd.Series:
            return self.raw_df.iloc[index]
        
        def get_numpy(self) -> tuple[np.ndarray]:
            '''
            Get the dataset in numpy format.
            
            Returns:
                - the dataset in numpy format (tuple[np.ndarray])
            '''
            X = self.raw_df.drop(columns=[self.target_column]).values
            y = self.raw_df[self.target_column].values
            return X, y 
        
        def get_raw_df(self) -> pd.DataFrame:
            '''
            Get the raw dataframe.
            
            Returns:
                - the raw dataframe (pd.DataFrame)
            '''
            return self.raw_df
        
        def overwrite_raw_df(self, new_raw_df: pd.DataFrame) -> None:
            '''
            Overwrite the raw dataframe.
            
            Parameters:
                - new_raw_df: the new raw dataframe (pd.DataFrame)
            '''
            self.raw_df = new_raw_df 
            self.data['raw_df'] = new_raw_df
            
        def get_target_column(self) -> str:
            '''
            Get the target column.
            
            Returns:
                - the target column (str)
            '''
            return self.target_column
        
        def get_continuous_columns(self) -> list[str]:
            '''
            Get the continuous columns.
            
            Returns:
                - the continuous columns (list[str])
            '''
            return self.continuous_columns

class DatasetPreprocessor:
    
    def __init__(self,
                dataset: Dataset,
                split: float = 0.8, 
                random_state: int = 0, 
                stratify: bool = False, 
                standardize_data: str = 'minmax',
                one_hot: bool = False,
                binarize_y: bool = True
                ) -> None:
        '''
        Initialize the dataset preprocessor.
        
        Parameters:
            - dataset: the dataset (Dataset)
            - split: the split of the dataset (float)
            - random_state: the random state (int)
            - stratify: whether to stratify the dataset (bool)
            - standardize_data: whether to standardize the dataset (str) should be one of ['minmax', 'zscore']
            - one_hot: whether to one-hot encode the dataset (bool)
            - binarize_y: whether to binarize the target variable (bool)
        '''
        
        self.dataset = dataset
        self.split = split
        self.random_state = random_state
        self.stratify = stratify
        self.standardize_data = standardize_data
        self.one_hot = one_hot
        self.binarize_y = binarize_y
        
        if self.standardize_data not in ['minmax', 'zscore']:
            raise ValueError('standardize_data should be one of ["minmax", "zscore"]')
        
        self.scaler = StandardScaler() if self.standardize_data == 'zscore' else MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False)
        self.label_encoder = LabelEncoder()
           
        self.raw_df = self.dataset.raw_df
        self.target_column = self.dataset.target_column
        self.continuous_columns = self.dataset.continuous_columns
        self.categorical_columns = self.dataset.categorical_columns
        
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
            
        if self.binarize_y:
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)
            
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
    
    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''
        Get the preprocessed data.
        
        Returns:
            - the preprocessed data (tuple[pd.DataFrame])
        '''
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_numpy(self) -> tuple[np.ndarray]:
        '''
        Get the preprocessed data in numpy format.
        
        Returns:
            - the preprocessed data in numpy format (tuple[np.ndarray])
        '''
        X_train = self.X_train.values
        X_test = self.X_test.values
        y_train = self.y_train.values
        y_test = self.y_test.values
        return X_train, X_test, y_train, y_test


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

def _moons(n_samples: int = 1000, noise: float = 0.1, random_state: int = 0) -> dict:
    '''
    Create two moons with the same number of samples and noise.
    
    Parameters:
        - n_samples: the number of samples for each moon (int)
        - noise: the noise of the moons (float)
        - random_state: the random state (int)
    '''
    data = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    data2 = make_moons(n_samples=n_samples, noise=noise, random_state=random_state + 1)
    
    X = np.concatenate([data[0], data2[0] / 1.5 + 1.6]) 
    y = np.concatenate([data[1], data2[1]])
    
    # Normalize the data to 0-1
    X = (X - X.min()) / (X.max() - X.min())
    
    raw_df = pd.DataFrame(X, columns=['x1', 'x2'])
    raw_df['y'] = y
    
    categorical_columns = []
    continuous_columns = ['x1', 'x2']
    target_column = 'y'
    freeze_columns = []
    feature_ranges = {
        'x1': [0, 1],
        'x2': [0, 1]
    }
    
    data = {
        'raw_df': raw_df,
        'categorical_columns': categorical_columns,
        'continuous_columns': continuous_columns,
        'target_column': target_column,
        'freeze_columns': freeze_columns,
        'feature_ranges': feature_ranges
    }
    
    return data

def _donuts(n_samples: int = 1000, noise: float = 0.1, random_state: int = 0) -> dict:
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
    
    raw_df = pd.DataFrame(X, columns=['x1', 'x2'])
    raw_df['y'] = y
    
    categorical_columns = []
    continuous_columns = ['x1', 'x2']
    target_column = 'y'
    freeze_columns = []
    feature_ranges = {
        'x1': [0, 1],
        'x2': [0, 1]
    }
    
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
    
    german_dataset = Dataset(name='moons')
    
    german_preprocessor = DatasetPreprocessor(german_dataset, standardize_data='minmax', one_hot=True)
    print(german_preprocessor.X_train.head())
    print(german_preprocessor.y_test)