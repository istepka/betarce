import numpy as np
import pandas as pd
from copy import deepcopy
import wandb

from scikit_models import load_model, __train_model, save_model
from sklearn.metrics import accuracy_score
from create_data_examples import DatasetPreprocessor, Dataset

class ExperimentDataBase: 
    def __init__(self) -> None:
        self.dataset: Dataset = None
        
    def create(self) -> None:
        raise NotImplementedError('Method not implemented.')
    
    def get_dataset(self) -> Dataset:
        return self.dataset

class TwoSamplesOneDatasetExperimentData(ExperimentDataBase):
    def __init__(self,
                 dataset: Dataset,
                 standardize: bool = False,
                 one_hot_encode: bool = False,
                 random_state: int | None = None) -> None:
        '''
        Parameters: 
            - dataset: (Dataset) The dataset to be used in the experiment.
            - standardize: (bool) Whether to standardize the data.
            - one_hot_encode: (bool) Whether to one-hot encode the data.
            - random_state: (int) Random state for reproducibility.
        '''
        super().__init__()
        self.dataset = dataset
        self.standardize = standardize
        self.one_hot_encode = one_hot_encode
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
    def __create_datasets(self) -> tuple[Dataset, Dataset]:
        '''
        Splits the dataset into two samples.
        
        Returns:
            - Tuple[Dataset, Dataset]: The two samples.
        '''
        
        dataset1 = deepcopy(self.dataset)
        dataset2 = deepcopy(self.dataset)
        
        l = len(dataset1)
        
        # Create a mask to split the dataset into two samples.
        mask = np.zeros(l, dtype=bool)
        mask = np.random.choice([True, False], size=l, replace=True)
        
        # Create the two samples
        raw_df = dataset1.get_raw_df()
        
        sample1 = raw_df[mask].reset_index(drop=True)
        sample2 = raw_df[~mask].reset_index(drop=True)
        
        dataset1.overwrite_raw_df(sample1)
        dataset2.overwrite_raw_df(sample2)    
        
        return dataset1, dataset2    
    
    def create(self) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
                              tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        '''
        Creates the two samples and preprocesses them.
        
        Returns:
            - The preprocessed samples. In form of (X_train, X_test, y_train, y_test) tuples for both samples.
        '''
        
        dataset1, dataset2 = self.__create_datasets()
        
        preprocessor1 = DatasetPreprocessor(dataset1,
                                            standardize_data=self.standardize,
                                            one_hot=self.one_hot_encode,
                                            random_state=self.random_state,
                                            split=0.8
        )
        
        preprocessor2 = DatasetPreprocessor(dataset2,
                                            standardize_data=self.standardize,
                                            one_hot=self.one_hot_encode,
                                            random_state=self.random_state,
                                            split=0.8
        )
        
        X_train1, X_test1, y_train1, y_test1 = preprocessor1.get_data()
        X_train2, X_test2, y_train2, y_test2 = preprocessor2.get_data()
        
        return (X_train1, X_test1, y_train1, y_test1), (X_train2, X_test2, y_train2, y_test2)

class ExperimentBase:
    def run(self) -> None:
        raise NotImplementedError('Method not implemented.')
    
class TwoDatasetsExperiment(ExperimentBase):
    
    def __init__(self, 
                 model_type: str,
                 experiment_data: TwoSamplesOneDatasetExperimentData,
                 wandb_logger: bool = False,
                 ) -> None:
        super().__init__()
        
        self.model_type = model_type
        
        self.experiment_data = experiment_data  
        self.wandb_logger = wandb_logger
        
        s1, s2 = self.experiment_data.create()
        self.X_train1, self.X_test1, self.y_train1, self.y_test1 = s1
        self.X_train2, self.X_test2, self.y_train2, self.y_test2 = s2
    
    def prepare(self) -> None: 
        # Fit two models on the two samples
        
        if self.model_type == 'mlp':
            hparams = {
            'hidden_layer_sizes': (100, 100),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.01,
            'max_iter': 1000,
            'random_state': 0,
            'verbose': True,
            'early_stopping': True,
            }
            
        elif self.model_type == 'rf':
            hparams = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 0,
            }
        else:
            raise ValueError('model_type must be either "mlp" or "rf"')
            
        self.model1 = __train_model(self.model_type, self.X_train1, self.y_train1, hparams=hparams)
        self.model2 = __train_model(self.model_type, self.X_train2, self.y_train2, hparams=hparams)
        
        save_model(self.model1, f'models/{self.model_type}_sample1_TwoDatasetsExperiment.joblib')
        save_model(self.model2, f'models/{self.model_type}_sample2_TwoDatasetsExperiment.joblib')
        
        # Calculate the accuracy of the models
        self.acc1 = accuracy_score(self.y_test1, self.model1.predict(self.X_test1))
        self.acc2 = accuracy_score(self.y_test2, self.model2.predict(self.X_test2))
        
        self.log_artifact('accuracy_model1', self.acc1)
        self.log_artifact('accuracy_model2', self.acc2)
        
    
    def run(self) -> None:
        # TODO: Implement the run method
        # 1) Generate counterfactual examples for all test samples of the first model
        # 2) Calculate metrics for model 1 and model 2
        pass
    
    def log_artifact(self, key: str, value: object) -> None:
        if self.wandb_logger:
            wandb.log({key: value})
        
        print(f'Accuracy of model 1: {self.acc1}')
        print(f'Accuracy of model 2: {self.acc2}')


if __name__ == '__main__':
    dataset = Dataset('german')
    
    e1 = TwoSamplesOneDatasetExperimentData(dataset, random_state=321)
    
    sample1, sample2 = e1.create()
    
    X_train1, X_test1, y_train1, y_test1 = sample1
    X_train2, X_test2, y_train2, y_test2 = sample2
    
    print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
    print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)
        