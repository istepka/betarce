import numpy as np
import pandas as pd
from copy import deepcopy
import wandb
from tqdm import tqdm
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
import warnings

from create_data_examples import DatasetPreprocessor, Dataset
from scikit_models import load_model, scikit_predict_proba_fn, train_model, save_model
from dice_wrapper import get_dice_explainer, get_dice_counterfactuals
from robx import robx_algorithm, counterfactual_stability

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
                 standardize: bool = True,
                 one_hot_encode: bool = True,
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
        self.target_column = experiment_data.dataset.get_target_column()
        self.continuous_features = experiment_data.dataset.get_continuous_columns()
        
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
        
        print(self.X_train1.head())
            
        self.model1 = train_model(self.model_type, self.X_train1, self.y_train1, hparams=hparams)
        self.model2 = train_model(self.model_type, self.X_train2, self.y_train2, hparams=hparams)
        
        
        save_model(self.model1, f'models/{self.model_type}_sample1_TwoDatasetsExperiment.joblib')
        save_model(self.model2, f'models/{self.model_type}_sample2_TwoDatasetsExperiment.joblib')
        
        # Calculate the accuracy of the models
        self.acc1 = accuracy_score(self.y_test1, self.model1.predict(self.X_test1))
        self.acc2 = accuracy_score(self.y_test2, self.model2.predict(self.X_test2))
        
        self.log_artifact('accuracy_model1', self.acc1)
        self.log_artifact('accuracy_model2', self.acc2)
        
    
    def run(self, base_cf_method: str = 'dice') -> dict:        
        data = [
            (self.model1, self.X_train1, self.X_test1, self.y_train1, self.y_test1),
            (self.model2, self.X_train2, self.X_test2, self.y_train2, self.y_test2)
        ]
        
        results = {
            'model1': {
                'original_instance': list(),
                'base_counterfactual': list(),
                'base_generation_time': list(),
                'base_validity': list(),
                'base_lof': list(),
                'base_proximity': list(),
                'base_counterfactual_stability': list(),
                'robust_counterfactual': list(),
                'robust_generation_time': list(),
                'robust_validity': list(),
                'robust_lof': list(),
                'robust_proximity': list(),
                'robust_counterfactual_stability': list(),
            },
            'model2': {
                'original_instance': list(),
                'base_counterfactual': list(),
                'base_generation_time': list(),
                'base_validity': list(),
                'base_lof': list(),
                'base_proximity': list(),
                'base_counterfactual_stability': list(),
                'robust_counterfactual': list(),
                'robust_generation_time': list(),
                'robust_validity': list(),
                'robust_lof': list(),
                'robust_proximity': list(),
                'robust_counterfactual_stability': list(),
            }
        }
        
        # 1) Generate counterfactual examples for all test samples of the first model
        for i, (model, X_train, X_test, y_train, y_test) in enumerate(data):
            
            results_key = f'model{i+1}'
            lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, p=1)
            lof_model.fit(X_train.to_numpy()) # Fit the LOF model without column names
            
            predict_dn = scikit_predict_proba_fn(model)
            
            
            X_train_w_target = X_train.copy()
            X_train_w_target[self.target_column] = y_train
            # Set up the explainer
            if base_cf_method == 'dice':
                explainer = get_dice_explainer(
                    dataset=X_train_w_target,
                    model=model,
                    outcome_name=self.target_column,
                    continous_features=X_train.columns.tolist(),
                    dice_method='kdtree',
                    # feature_encoding='ohe-min-max'
                )
            else:
                raise ValueError('base_cf_method must be "dice"')
            
                
            warnings.filterwarnings('ignore', category=UserWarning)
            for j in tqdm(range(len(X_test)), total=len(X_test)):
                
                x = X_test[j:j+1] # Get the instance to be explained pd.DataFrame
                y = y_test[j] # Get the true label
                
                start_time = time.time()
                dice_cf = get_dice_counterfactuals(
                    dice_exp=explainer,
                    query_instance=x,
                    total_CFs=1,
                    desired_class='opposite',
                    proximity_weight=1.0,
                    diversity_weight=0.5,
                )
                
                generation_time = time.time() - start_time # Get the generation time in seconds
                if dice_cf is None:
                    cf_numpy = None
                else: 
                    cf_numpy = dice_cf[0].final_cfs_df.to_numpy()[0][:-1]
                x_numpy = x.to_numpy()[0]
                cf_label = model.predict(cf_numpy.reshape(1, -1))[0]
                validity = int(cf_label) == 1 - int(y) # Opposite class is valid
                proximityL1 = np.sum(np.abs(x_numpy - cf_numpy))
                lof = lof_model.predict(cf_numpy.reshape(1, -1))[0]
                cf_counterfactual_stability = counterfactual_stability(
                    cf_numpy, predict_dn, 0.1, 1000
                )
                
                results[results_key]['original_instance'].append(x_numpy)
                results[results_key]['base_counterfactual'].append(cf_numpy)
                results[results_key]['base_generation_time'].append(generation_time)
                results[results_key]['base_validity'].append(validity)
                results[results_key]['base_lof'].append(lof)
                results[results_key]['base_proximity'].append(proximityL1)
                results[results_key]['base_counterfactual_stability'].append(cf_counterfactual_stability)
                
                
                start_time = time.time()
                robust_cf = robx_algorithm(
                    X_train = X_train.to_numpy(),
                    predict_class_proba_fn = predict_dn,
                    start_counterfactual = x_numpy,
                    tau = 0.93,
                    variance = 0.1,
                    N = 1000,
                )
                
                generation_time = time.time() - start_time # Get the generation time in seconds
                
                if robust_cf is None:
                    cf_numpy = None
                    cf_label = None
                    validity = None
                    proximityL1 = None
                    lof = None
                    cf_counterfactual_stability = None
                else:
                    cf_numpy = robust_cf
                    cf_label = model.predict(cf_numpy.reshape(1, -1))[0]
                    validity = int(cf_label) == 1 - int(y) # Opposite class is valid
                    proximityL1 = np.sum(np.abs(x_numpy - cf_numpy))
                    lof = lof_model.predict(cf_numpy.reshape(1, -1))[0]
                    cf_counterfactual_stability = counterfactual_stability(
                        cf_numpy, predict_dn, 0.1, 1000
                    )
                
                results[results_key]['robust_counterfactual'].append(cf_numpy)
                results[results_key]['robust_generation_time'].append(generation_time)
                results[results_key]['robust_validity'].append(validity)
                results[results_key]['robust_lof'].append(lof)
                results[results_key]['robust_proximity'].append(proximityL1)
                results[results_key]['robust_counterfactual_stability'].append(cf_counterfactual_stability)
                
                
                
                
                if j == 3: # WARNING: REMOVE THIS LINE AFTER TESTING
                    break
            
        return results
            
            
        
        # 2) Calculate metrics for model 1 and model 2
        pass
    
    def log_artifact(self, key: str, value: object) -> None:
        if self.wandb_logger:
            wandb.log({key: value})
        
        print(f'Accuracy of model 1: {self.acc1}')
        print(f'Accuracy of model 2: {self.acc2}')


if __name__ == '__main__':
    dataset = Dataset('german')
    
    e1 = TwoSamplesOneDatasetExperimentData(
        dataset, 
        random_state=22,
        one_hot_encode=True,
        )
    
    sample1, sample2 = e1.create()
    
    X_train1, X_test1, y_train1, y_test1 = sample1
    X_train2, X_test2, y_train2, y_test2 = sample2
    
    print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
    print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)
    
    
    td_exp = TwoDatasetsExperiment('mlp', e1)
    td_exp.prepare()
    res = td_exp.run()
    
    print(res)
        