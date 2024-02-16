import os 
print(os.getcwd())

from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
import wandb
from tqdm import tqdm
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
import warnings
import joblib

from create_data_examples import DatasetPreprocessor, Dataset
from scikit_models import load_model, scikit_predict_proba_fn, train_model, save_model, train_calibrated_model
from dice_wrapper import get_dice_explainer, get_dice_counterfactuals
from robx import robx_algorithm, counterfactual_stability
from config_wrapper import ConfigWrapper
from explainers import BaseExplainer, DiceExplainer, AlibiWachter

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
                 standardize: str = 'minmax',
                 one_hot_encode: bool = True,
                 random_state: int | None = None) -> None:
        '''
        Parameters: 
            - dataset: (Dataset) The dataset to be used in the experiment.
            - standardize: (str) The standardization method. Either 'minmax' or 'zscore'.
            - one_hot_encode: (bool) Whether to one-hot encode the data.
            - random_state: (int) Random state for reproducibility.
        '''
        super().__init__()
        self.dataset = dataset
        self.standardize = standardize
        self.one_hot_encode = one_hot_encode
        self.random_state = random_state
        
        if standardize not in ['minmax', 'zscore']:
            raise ValueError('standardize must be either "minmax" or "zscore"')
        
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
        
        self.preprocessor1 = DatasetPreprocessor(dataset1,
                                            standardize_data=self.standardize,
                                            one_hot=self.one_hot_encode,
                                            random_state=self.random_state,
                                            split=0.85
        )
        
        self.preprocessor2 = DatasetPreprocessor(dataset2,
                                            standardize_data=self.standardize,
                                            one_hot=self.one_hot_encode,
                                            random_state=self.random_state,
                                            split=0.85
        )
        
        X_train1, X_test1, y_train1, y_test1 = self.preprocessor1.get_data()
        X_train2, X_test2, y_train2, y_test2 = self.preprocessor2.get_data()
        
        return (X_train1, X_test1, y_train1, y_test1), (X_train2, X_test2, y_train2, y_test2)

class ExperimentResults:
    def __init__(self) -> None:
        self.results = defaultdict(list)
        self.artifacts = {}
    
    def add_metric(self, metric: str, value: float) -> None:
        self.results[metric].append(value)
        
    def add_artifact(self, key: str, value: object) -> None:
        self.artifacts[key] = value
    
    def get_artifact(self, key: str) -> object:
        return self.artifacts[key]
    
    def get_all_artifacts(self) -> dict:
        return self.artifacts
    
    def get_all_artifact_keys(self) -> list:
        return list(self.artifacts.keys())
    
    def get_all_results(self) -> dict:
        return self.results
    
    def get_mean_results(self) -> dict:
        return {k: np.mean(v) for k, v in self.results.items()}
    
    def get_std_results(self) -> dict:
        return {k: np.std(v) for k, v in self.results.items()}
    
    def get_results_for_metric(self, metric: str) -> list:
        return self.results[metric]
    
    def get_mean_for_metric(self, metric: str) -> float:
        return np.mean(self.results[metric])
    
    def get_std_for_metric(self, metric: str) -> float:
        return np.std(self.results[metric])
    
    def reset(self) -> None:
        self.results = defaultdict(lambda: list())
        
    def get_all_metric_names(self) -> list:
        return list(self.results.keys())
    
    def get_results_as_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
    
    def __str__(self) -> str:
        return f'ExperimentResults with {len(self.results)} metrics and {len(self.artifacts)} artifacts.'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def pretty_print(self) -> None:
        print(self.__str__())
        print('Metrics:')
        for k, v in self.results.items():
            print(f'{k}: {self.get_mean_for_metric(k):.2f} (std: {self.get_std_for_metric(k):.2f})')
            
    def pretty_print_robust_vs_base(self) -> None:
        print(self.__str__())
        print('#' * 30 + ' Metrics ' + '#' * 30)
        # robust_metrics = [k for k in self.results.keys() if 'robust' in k]
        r_2 = [k for k in self.results.keys() if 'robust' in k and '2' in k]
        r_1 = [k for k in self.results.keys() if 'robust' in k and '2' not in k]
        # base_metrics = [k for k in self.results.keys() if 'robust' not in k]
        b_2 = [k for k in self.results.keys() if 'robust' not in k and '2' in k]
        b_1 = [k for k in self.results.keys() if 'robust' not in k and '2' not in k]
        print('-' * 25 + ' Base metrics ' + '-' * 25)
        for k in b_1:
            print(f'{k}: {self.get_mean_for_metric(k):.2f} (std: {self.get_std_for_metric(k):.2f})')
        print('-' * 25 + ' Base metrics 2 ' + '-' * 25)
        for k in b_2:
            print(f'{k}: {self.get_mean_for_metric(k):.2f} (std: {self.get_std_for_metric(k):.2f})')
        print('-' * 25 + ' Robust metrics ' + '-' * 25)
        for k in r_1:
            print(f'{k}: {self.get_mean_for_metric(k):.2f} (std: {self.get_std_for_metric(k):.2f})')
        print('-' * 25 + ' Robust metrics 2 ' + '-' * 25)
        for k in r_2:
            print(f'{k}: {self.get_mean_for_metric(k):.2f} (std: {self.get_std_for_metric(k):.2f})')
        print('#' * 80)
        
    def save_to_file(self, path: str) -> bool | None:
        try:
            with open(path, 'wb') as f:
                joblib.dump(self, f)
            return True
        except Exception as e:
            print(f'Error serializing results: {e}')
            return None
        
    @staticmethod
    def load_results_from_file(path: str) -> object | None:
        try: 
            with open(path, 'rb') as f:
                return joblib.load(f)
        except Exception as e:
            print(f'Error loading results from file: {e}')
            return None    

class ExperimentBase:
    def run(self) -> None:
        raise NotImplementedError('Method not implemented.')

class TwoDatasetsExperiment(ExperimentBase):
    
    def __init__(self, 
                 model_type: str,
                 experiment_data: TwoSamplesOneDatasetExperimentData,
                 config: ConfigWrapper | None = None,
                 wandb_logger: bool = False,
                 custom_experiment_name: str | None = None
                 ) -> None:
        super().__init__()
        
        self.model_type = model_type
        self.config = config
        self.custom_experiment_name = custom_experiment_name \
            if custom_experiment_name is not None else 'TwoDatasetsExperiment'
        
        self.experiment_data = experiment_data  
        self.wandb_logger = wandb_logger
        self.target_column = experiment_data.dataset.get_target_column()
        self.continuous_features = experiment_data.dataset.get_continuous_columns()
        
        self.class_threshold = self.config.get_config_by_key('classification_threshold')
        
        s1, s2 = self.experiment_data.create()
        self.X_train1, self.X_test1, self.y_train1, self.y_test1 = s1
        self.X_train2, self.X_test2, self.y_train2, self.y_test2 = s2
        
        self.robXHparams = config.get_model_config('robXHparams')
        
        self.results = ExperimentResults()
    
    def prepare(self, 
            calibrate: bool = False,
            calibrate_method: str = 'isotonic'
        ) -> None: 
        '''
        Prepares the experiment by training the models and calculating the accuracy.
        
        Parameters:
            - calibrate: (bool) Whether to calibrate the models.
            - calibrate_method: (str) The calibration method. Either 'isotonic' or 'sigmoid'.
        '''
        # Fit two models on the two samples  
        if self.model_type == 'mlp':
            hparams = self.config.get_config_by_key('mlp')
        elif self.model_type == 'rf':
            hparams = self.config.get_config_by_key('rf')
        else:
            raise ValueError('model_type must be either "mlp" or "rf"')
        
        print(self.X_train1.head())
        print(hparams)
            
        if calibrate:
            self.model1 = train_calibrated_model(self.model_type, self.X_train1, self.y_train1, 
                                                 base_model_hparams=hparams, calibration_method=calibrate_method)
        else:
            self.model1 = train_model(self.model_type, self.X_train1, self.y_train1, hparams=hparams)
            
        self.model2 = train_model(self.model_type, self.X_train2, self.y_train2, hparams=hparams)
        
        save_model(self.model1, f'models/{self.custom_experiment_name}_sample1.joblib')
        save_model(self.model2, f'models/{self.custom_experiment_name}_sample2.joblib')
        
        # Calculate the accuracy of the models
        self.acc1 = accuracy_score(self.y_test1, self.model1.predict(self.X_test1))
        self.acc2 = accuracy_score(self.y_test2, self.model2.predict(self.X_test2))
        
        self.log_artifact('accuracy_model1', self.acc1)
        self.log_artifact('accuracy_model2', self.acc2)
           
    def run(self, 
            base_cf_method: str = 'dice',
            stop_after: int | None = None
        ) -> bool:
        '''
        Runs the experiment.
        
        Parameters:
            - base_cf_method: (str) The method to be used for generating counterfactuals. Either 'dice' or 'wachter'.
            - stop_after: (int) The number of iterations to run the experiment. If None, runs for all test samples.
        '''
        # 1) Generate counterfactual examples for all test samples of the first model
        X_train = self.X_train1
        X_test = self.X_test1
        y_train = self.y_train1
        y_test = self.y_test1
        model = self.model1
        
        lof_model = LocalOutlierFactor(n_neighbors=50, novelty=True, p=1)
        lof_model.fit(X_train.to_numpy()) # Fit the LOF model without column names
        lof_model2 = LocalOutlierFactor(n_neighbors=50, novelty=True, p=1)
        lof_model2.fit(self.X_train2.to_numpy())
        
        predict_fn = scikit_predict_proba_fn(model)
        predict_fn_2 = scikit_predict_proba_fn(self.model2)
        
        empty_metrics = {
            'validity': np.nan,
            'proximityL1': np.nan,
            'lof': np.nan,
            'cf_counterfactual_stability': np.nan
        }
        
        X_train_w_target = X_train.copy()
        X_train_w_target[self.target_column] = y_train
        X_train_w_target = X_train_w_target.reset_index(drop=True).copy()
        
        # Set up the explainer
        match base_cf_method:
            case 'dice':
                explainer = DiceExplainer(
                    dataset=X_train_w_target,
                    model=model,
                    outcome_name=self.target_column,
                    continous_features=self.continuous_features
                )
                explainer.prep(
                    dice_method='random',
                    feature_encoding=None
                )
            case 'wachter':
                import tensorflow as tf
                tf.compat.v1.disable_eager_execution()
                explainer = AlibiWachter(
                    model=model,
                    dataset=X_train_w_target,
                    outcome_name=self.target_column,
                    continuous_features=self.continuous_features
                )
                shape = X_train.iloc[0:1].to_numpy().shape
                assert len(shape) == 2, 'The shape of the query instance must be (1, n_features)'
                
                predict_fn_for_wachter = lambda x: model.predict_proba(x)
                
                explainer.prep(
                    query_instance_shape=shape,
                    pred_fn=predict_fn_for_wachter,
                )
            case _:
                raise ValueError('base_cf_method must be either "dice" or "wachter"')
            
        warnings.filterwarnings('ignore', category=UserWarning)
        for j in tqdm(range(len(X_test)), total=len(X_test)):
            
            orig_x = X_test[j:j+1] # Get the instance to be explained pd.DataFrame
            orig_y = int(model.predict(orig_x)[0]) # Get the label of the instance, as we rely on the model not on the ground truth
            
            start_time = time.time()
            try:
                match base_cf_method:
                    case 'dice':
                        base_cf = explainer.generate(
                            query_instance=orig_x,
                            total_CFs=1,
                            desired_class= 1 - orig_y,
                            classification_threshold=self.class_threshold,
                        )
                    case 'wachter':
                        base_cf = explainer.generate(
                            query_instance=orig_x,
                        )
                    case _:
                        raise ValueError('base_cf_method must be "dice" or "wachter"')       
            except Exception as e:
                base_cf = None
                print(f'Error generating counterfactual: {e}')
            generation_time = time.time() - start_time # Get the generation time in seconds
            
            x_numpy = orig_x.to_numpy()[0]
            
            if base_cf is None:
                cf_numpy = None
                metrics = empty_metrics.copy()
                metrics_2 = empty_metrics.copy()
            else: 
                cf_numpy = base_cf
                
                metrics = self.calculate_metrics(
                    cf = cf_numpy,
                    cf_desired_class=1 - orig_y,
                    x = x_numpy,
                    model = model,
                    lof_model = lof_model,
                    predict_fn = predict_fn,
                    robXHparams = self.robXHparams
                )
                  
                metrics_2 = self.calculate_metrics(
                    cf = cf_numpy,
                    cf_desired_class=1 - orig_y,
                    x = x_numpy,
                    model = self.model2,
                    lof_model = lof_model2,
                    predict_fn = predict_fn_2,
                    robXHparams = self.robXHparams
                )    
                
            for k, v in metrics.items():
                self.results.add_metric(k, v)
            for k, v in metrics_2.items():
                self.results.add_metric(f'{k}_2', v)
                
            self.results.add_artifact('original_instance', x_numpy)
            self.results.add_artifact('base_counterfactual', cf_numpy)
            
            self.results.add_metric('generation_time', generation_time)
                
                
            # ROBX PART      
            start_time = time.time()
            try:
                robust_cf, _ = robx_algorithm(
                    X_train = X_train.to_numpy(),
                    predict_class_proba_fn = predict_fn,
                    start_counterfactual = cf_numpy,
                    tau = self.robXHparams['tau'],
                    variance = self.robXHparams['variance'],
                    N = self.robXHparams['N'],
                )
            except Exception as e:
                robust_cf = None
                print(f'Error generating robust counterfactual: {e}')
            rob_generation_time = time.time() - start_time # Get the generation time in seconds
            
            if robust_cf is None:
                rob_metrics = empty_metrics.copy()
                rob_metrics_2 = empty_metrics.copy()
            else:    
                cf_desired_class = 1 - orig_y
                
                rob_metrics = self.calculate_metrics(
                    cf = robust_cf,
                    cf_desired_class=cf_desired_class,
                    x = x_numpy,
                    model = model,
                    lof_model = lof_model,
                    predict_fn = predict_fn,
                    robXHparams = self.robXHparams
                )
            
                rob_metrics_2 = self.calculate_metrics(
                    cf = robust_cf,
                    cf_desired_class=cf_desired_class,
                    x = x_numpy,
                    model = self.model2,
                    lof_model = lof_model2,
                    predict_fn = predict_fn_2,
                    robXHparams = self.robXHparams
                )
                
            for k, v in rob_metrics.items():
                self.results.add_metric(f'robust_{k}', v)
                
            for k, v in rob_metrics_2.items():
                self.results.add_metric(f'robust_{k}_2', v)
            
            self.results.add_artifact('robust_counterfactual', robust_cf)
            self.results.add_metric('robust_generation_time', rob_generation_time)
            
            if stop_after and j >= stop_after:
                print(f'Stopping after {stop_after} iterations.')
                break
            
        return True
    
    def calculate_metrics(self, cf: np.ndarray, 
                          cf_desired_class: int,
                          x: np.ndarray, 
                          model: object,
                          lof_model: LocalOutlierFactor,
                          predict_fn: callable,
                          robXHparams: dict
        ) -> dict:
        '''
        Calculates the metrics for a counterfactual example.
        
        Parameters:
            - cf: (np.ndarray) The counterfactual example.
            - cf_desired_class: (int) The desired class of the counterfactual example.
            - x: (np.ndarray) The original instance.
            - model: (object) The model to be used for predictions.
            - lof_model: (object) The Local Outlier Factor model.
            - predict_fn: (function) The function to be used for predictions.
            - robXHparams: (dict) The hyperparameters for the RobustX algorithm.
        
        Returns:
            - dict: The metrics.
        '''
        
        cf_label = predict_fn(cf)[0] > self.class_threshold
        validity = int(cf_label) == cf_desired_class
        proximityL1 = np.sum(np.abs(x - cf))
        lof = lof_model.score_samples(cf.reshape(1, -1))[0]
        cf_counterfactual_stability = counterfactual_stability(
            cf, predict_fn, robXHparams['variance'], robXHparams['N']
        )
        
        return {
            'validity': validity,
            'proximityL1': proximityL1,
            'lof': lof,
            'cf_counterfactual_stability': cf_counterfactual_stability
        }
    
    def log_artifact(self, key: str, value: object) -> None:
        if self.wandb_logger:
            wandb.log({key: value})
        
        print(f'Accuracy of model 1: {self.acc1}')
        print(f'Accuracy of model 2: {self.acc2}')
           
    def get_results(self) -> ExperimentResults:
        return self.results
        

if __name__ == '__main__':
    config_wrapper = ConfigWrapper('config.yml')
    
    np.random.seed(config_wrapper.get_config_by_key('random_state'))
    results_dir = config_wrapper.get_config_by_key('result_path')
    
    experiments = [
        {
            'model_type': 'mlp',
            'base_cf_method': 'dice',
            'calibrate': False,
            'calibrate_method': None,
            'custom_experiment_name': 'mlp_base'
        },
        {
            'model_type': 'mlp',
            'base_cf_method': 'dice',
            'calibrate': True,
            'calibrate_method': 'isotonic',
            'custom_experiment_name': 'mlp_isotonic-cv'
        },
        {
            'model_type': 'mlp',
            'base_cf_method': 'dice',
            'calibrate': True,
            'calibrate_method': 'sigmoid',
            'custom_experiment_name': 'mlp_sigmoid-cv'
        }
    ]
    
    
    def __run_experiment(exp_config: dict, rep: int):
        _exp = deepcopy(exp_config)
        _exp['custom_experiment_name'] = f"{_exp['custom_experiment_name']}_{rep}"
        
        # Set the random state for reproducibility, by adding the rep number to the seed
        _config_wrapper = config_wrapper.copy()
        seed = _config_wrapper.get_config_by_key('random_state') + rep
        seed = int(seed)
        _config_wrapper.set_config_by_key('random_state', seed)
        
        dataset = Dataset('german')
        
        e1 = TwoSamplesOneDatasetExperimentData(
            dataset, 
            random_state=seed,
            one_hot_encode=True,
            standardize='minmax'
            )
        
        sample1, sample2 = e1.create()
        
        X_train1, X_test1, y_train1, y_test1 = sample1
        X_train2, X_test2, y_train2, y_test2 = sample2
        
        print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
        print(X_train2.shape, X_test2.shape, y_train2.shape, y_test2.shape)
        
        td_exp = TwoDatasetsExperiment(
            model_type=_exp['model_type'],
            experiment_data=e1,
            config=_config_wrapper,
            wandb_logger=False,
            custom_experiment_name=_exp['custom_experiment_name']
        )
        
        td_exp.prepare(
            calibrate=_exp['calibrate'],
            calibrate_method=_exp['calibrate_method']
        )
        
        run_status = td_exp.run(
            base_cf_method=_exp['base_cf_method'],
            stop_after=None
        )
        
        results = td_exp.get_results()
        results.save_to_file(f'{results_dir}/{_exp["custom_experiment_name"]}.joblib')
        
    
    import multiprocessing as mp
    
    
    for exp_config in experiments:
        
        processes = []
        for rep in range(10):
            
            p = mp.Process(target=__run_experiment, args=(exp_config, rep))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    print('All experiments finished.')
    
    
            