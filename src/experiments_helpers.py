from abc import abstractmethod
import os
from typing import Union

from sklearn.model_selection import train_test_split 
print(os.getcwd())

from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
import wandb
from tqdm import tqdm
import time
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import accuracy_score
import warnings
import joblib

from create_data_examples import DatasetPreprocessor, Dataset
from scikit_models import load_model, scikit_predict_proba_fn, train_model, save_model, train_calibrated_model, scikit_predict_crisp_fn
from dice_wrapper import get_dice_explainer, get_dice_counterfactuals
from robx import robx_algorithm, counterfactual_stability
from config_wrapper import ConfigWrapper
from explainers import BaseExplainer, DiceExplainer, GrowingSpheresExplainer
from statrob import StatrobGlobal, MLPClassifier, StatRobXPlus

class ExperimentDataBase: 
    def __init__(self,
                 dataset: Dataset,
                 standardize: str = 'minmax',
                 one_hot_encode: bool = True,
                 random_state: int | None = None,
                 train_test_split_ratio: float = 0.85
        ) -> None:
        '''
        Parameters: 
            - dataset: (Dataset) The dataset to be used in the experiment.
            - standardize: (str) The standardization method. Either 'minmax' or 'zscore'.
            - one_hot_encode: (bool) Whether to one-hot encode the data.
            - random_state: (int) Random state for reproducibility.
            - train_test_split_ratio: (float) The ratio of the train-test split.
        '''
        super().__init__()
        self.dataset = dataset
        self.standardize = standardize
        self.one_hot_encode = one_hot_encode
        self.random_state = random_state
        self.train_test_split_ratio = train_test_split_ratio
        
        if standardize not in ['minmax', 'zscore']:
            raise ValueError('standardize must be either "minmax" or "zscore"')
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def _create_datasets(self) -> tuple[Dataset, Dataset]:
        raise NotImplementedError('Method not implemented.')
        
    def create(self) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
                              tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        '''
        Creates the two samples and preprocesses them.
        
        Returns:
            - The preprocessed samples. In form of (X_train, X_test, y_train, y_test) tuples for both samples.
        '''
        
        dataset1, dataset2 = self._create_datasets()
        
        self.preprocessor = DatasetPreprocessor(dataset1,
                                            standardize_data=self.standardize,
                                            one_hot=self.one_hot_encode,
                                            random_state=self.random_state,
                                            split=self.train_test_split_ratio
        )
        
        self.preprocessor2 = DatasetPreprocessor(dataset2,
                                            standardize_data=self.standardize,
                                            one_hot=self.one_hot_encode,
                                            random_state=self.random_state,
                                            split=self.train_test_split_ratio
        )
        
        X_train1, X_test1, y_train1, y_test1 = self.preprocessor.get_data()
        X_train2, X_test2, y_train2, y_test2 = self.preprocessor2.get_data()
        
        return (X_train1, X_test1, y_train1, y_test1), (X_train2, X_test2, y_train2, y_test2)
    
    def get_dataset(self) -> Dataset:
        return self.dataset


class TwoSamplesOneDatasetExperimentData(ExperimentDataBase):
    def _create_datasets(self) -> tuple[Dataset, Dataset]:
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
    
class SameSampleExperimentData(ExperimentDataBase):
    def _create_datasets(self) -> tuple[Dataset, Dataset]:
        '''
        Creates two identical samples.
        
        Returns:
            - Tuple[Dataset, Dataset]: The two samples.
        '''
        dataset1 = deepcopy(self.dataset)
        dataset2 = deepcopy(self.dataset)
        
        return dataset1, dataset2

    
class ExperimentResults:
    def __init__(self, expected_size: int | None = None) -> None:
        self.results = defaultdict(list)
        self.records = defaultdict(list)
        self.artifacts = defaultdict(list)
        self.wandb_run = wandb.run

        
    def add_metric(self, metric: str, value: float) -> None:
        self.results[metric].append(value)
        
        if self.wandb_run:
            if isinstance(value, bool):
                value = int(value)
            wandb.log({metric: value})
    
    def add_artifact(self, key: str, value: object) -> None:
        self.artifacts[key].append(value)
        
        if self.wandb_run:
            art = wandb.Artifact(
                key,
                type="data_example",
                metadata={
                    "object": value,
                },
            )
            wandb.log_artifact(art)
            
    def add_record(self, key: str, value: object) -> None:
        self.records[key].append(value)
    
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
        r_2 = [k for k in self.results.keys() if 'robust' in k and '_2' in k]
        r_1 = [k for k in self.results.keys() if 'robust' in k and '_2' not in k]
        # base_metrics = [k for k in self.results.keys() if 'robust' not in k]
        b_2 = [k for k in self.results.keys() if 'robust' not in k and '_2' in k]
        b_1 = [k for k in self.results.keys() if 'robust' not in k and '_2' not in k]
        # The rest of the metrics
        rest = [k for k in self.results.keys() if k not in r_2 + r_1 + b_2 + b_1]
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
        print('-' * 25 + ' Rest of the metrics ' + '-' * 25)
        for k in rest:
            print(f'{k}: {self.get_mean_for_metric(k):.2f} (std: {self.get_std_for_metric(k):.2f})')
        print('#' * 80)
        
    def save_to_file(self, path: str) -> bool | None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'wb') as f:
                joblib.dump(self, f)
                  
            if self.wandb_run:
                artifact = wandb.Artifact(
                    'experiment_results',
                    type="ExperimentResults",
                )
                artifact.add_file(path)
                wandb.log_artifact(artifact)
                
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


    def add_experiment_results(self, other) -> None:
        '''
        Adds the results of another experiment to this one.
        
        Parameters:
            - other: (ExperimentResults) The other experiment.
        '''
        assert isinstance(other, ExperimentResults), f'other must be an ExperimentResults object. Got {type(other)}'
        
        for k, v in other.get_all_results().items():
            self.results[k].extend(v)
            
        for k, v in other.get_all_artifacts().items():
            self.artifacts[k].extend(v)
            
        for k, v in other.records.items():
            self.records[k].extend(v)

class ExperimentBase:
    def __init__(self) -> None:
        self.model_type: str 
        self.config: ConfigWrapper
        self.custom_experiment_name: str
        
        self.experiment_data: ExperimentDataBase
        self.wandb_logger: wandb.run
        self.target_column: str
        self.continuous_features: list[str]
        
        self.class_threshold: float
        
        self.X_train1: pd.DataFrame
        self.X_test1: pd.DataFrame
        self.y_train1: pd.Series
        self.y_test1: pd.Series
        
        self.X_train2: pd.DataFrame
        self.X_test2: pd.DataFrame
        self.y_train2: pd.Series
        self.y_test2: pd.Series
            
        self.preprocessor: DatasetPreprocessor
        self.robXHparams: dict
        
        self.results: ExperimentResults = ExperimentResults()
        self.prep_done: bool = False
        
        self.lof_model: LocalOutlierFactor = LocalOutlierFactor(n_neighbors=50, novelty=True, p=1)
        self.lof_model2: LocalOutlierFactor = LocalOutlierFactor(n_neighbors=50, novelty=True, p=1)
        
        self.nearest_neighbors: NearestNeighbors = NearestNeighbors(n_neighbors=15, algorithm='ball_tree')
        self.nearest_neighbors2: NearestNeighbors = NearestNeighbors(n_neighbors=15, algorithm='ball_tree')
        
        
        
        
        
        
    def prepare(self):
        raise NotImplementedError('Method not implemented.')
        
    def run(self, 
            robust_method: str, 
            base_cf_method: str,
            stop_after: int | None = None
        ) -> bool:
        '''
        Runs the experiment.
        
        Parameters:
            - robust_method: (str) The method to be used for generating robust counterfactuals. Either 'robx' or 'statrob'.
            - base_cf_method: (str) The method to be used for generating counterfactuals. Either 'dice' or 'wachter' or 'gs'.
            - stop_after: (int) The number of iterations to run the experiment. If None, runs for all test samples.
        '''
        assert robust_method in ['robx', 'statrob', 'statrobxplus'], 'robust_method must be either "robx" or "statrob" or "statrobxplus"'
        assert base_cf_method in ['dice', 'wachter', 'gs'], 'base_cf_method must be either "dice" or "wachter" or "gs"'
        assert self.prep_done, 'prepare() must be called before run()'
        
        # 1) Generate counterfactual examples for all test samples of the first model
        X_train = self.X_train1
        X_test = self.X_test1
        y_train = self.y_train1
        y_test = self.y_test1
        model = self.model1
        
        # Prepare the results object, with the expected size, which will speed up the process
        self.results = ExperimentResults(expected_size = stop_after if stop_after else len(X_test))
        
        
        self.lof_model.fit(self.X_train1.to_numpy())
        self.lof_model2.fit(self.X_train2.to_numpy())
        self.nearest_neighbors.fit(self.X_train1.to_numpy())
        self.nearest_neighbors2.fit(self.X_train2.to_numpy())
        
        empty_metrics = {
            'validity': np.nan,
            'proximityL1': np.nan,
            'proximityL2': np.nan,
            'lof': np.nan,
            'cf_counterfactual_stability': np.nan,
            'dpow': np.nan,
            'plausibility': np.nan
        }
        
        X_train_w_target = X_train.copy()
        X_train_w_target[self.target_column] = y_train
        X_train_w_target = X_train_w_target.reset_index(drop=True).copy()
        
        # Set up the base explainer method
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
                from explainers import AlibiWachter
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
            case 'gs':
                _gsconfig = self.config.get_config_by_key('statrobHparams')['growingSpheresHparams']
                
                explainer = GrowingSpheresExplainer(
                    keys_mutable=self.preprocessor.X_train.columns.tolist(),
                    keys_immutable=[],
                    feature_order=self.preprocessor.X_train.columns.tolist(),
                    binary_cols=self.preprocessor.encoder.get_feature_names_out().tolist(),
                    continous_cols=self.preprocessor.continuous_columns,
                    pred_fn_crisp=self.predict_fn_1_crisp,
                    target_proba=_gsconfig['target_proba'],
                    max_iter=_gsconfig['max_iter'],
                    n_search_samples=_gsconfig['n_search_samples'],
                    p_norm=_gsconfig['p_norm'],
                    step=_gsconfig['step']
                )    
                explainer.prep()
            case _:
                raise ValueError('base_cf_method must be either "dice" or "wachter" or "gs"')
         
        # Set up robust explainer
        match robust_method:
            case 'robx':
                pass
            case 'statrob':
                statrobExplainer = StatrobGlobal(
                    dataset=X_train.to_numpy(),
                    preprocessor=self.preprocessor,
                    blackbox=model,
                    seed=self.config.get_config_by_key('random_state'),
                )
                
                statrobExplainer.fit(k_mlps=self.config.get_config_by_key('statrobHparams')['k_mlps'])
            case 'statrobxplus':
                statrobxplusExplainer = StatRobXPlus(
                    dataset=X_train.to_numpy(),
                    preprocessor=self.preprocessor,
                    blackbox=model,
                    seed=self.config.get_config_by_key('random_state'),
                )
                
                statrobxplusExplainer.fit(k_mlps=self.config.get_config_by_key('statrobHparams')['k_mlps'])        
            
            case _:
                raise ValueError('robust_method must be either "robx" or "statrob"')
            
        warnings.filterwarnings('ignore', category=UserWarning)
        for j in tqdm(range(len(X_test)), total=len(X_test), desc='Running experiment'):
            
            orig_x = X_test[j:j+1] # Get the instance to be explained pd.DataFrame
            orig_y = int(self.predict_fn_1_crisp(orig_x)[0]) # Get the label of the instance, as we rely on the model not on the ground truth
            
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
                        print('DICE')
                    case 'wachter':
                        base_cf = explainer.generate(
                            query_instance=orig_x,
                        )
                    case 'gs':
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
            
            if base_cf is None or 'nan' in base_cf.astype(str):
                cf_numpy = None
                metrics = empty_metrics.copy()
                metrics_2 = empty_metrics.copy()
            else: 
                cf_numpy = base_cf
                
                metrics = self.calculate_metrics(
                    cf = cf_numpy,
                    cf_desired_class=1 - orig_y,
                    x = x_numpy,
                    X_train=X_train.to_numpy(),
                    y_train=y_train,
                    lof_model = self.lof_model,
                    nearest_neighbors_model=self.nearest_neighbors,
                    predict_fn = self.predict_fn_1,
                    robXHparams = self.robXHparams
                )
                  
                metrics_2 = self.calculate_metrics(
                    cf = cf_numpy,
                    cf_desired_class=1 - orig_y,
                    x = x_numpy,
                    X_train=self.X_train2.to_numpy(),
                    y_train=self.y_train2,
                    lof_model = self.lof_model2,
                    nearest_neighbors_model=self.nearest_neighbors2,
                    predict_fn = self.predict_fn_2,
                    robXHparams = self.robXHparams
                )    
                
            for k, v in metrics.items():
                self.results.add_metric(k, v)
            for k, v in metrics_2.items():
                self.results.add_metric(f'{k}_2', v)
                
            self.results.add_metric('generation_time', generation_time)
                
            # FIND ROBUST COUNTERFACTUAL    
            start_time = time.time()
            try:
                match robust_method:
                    case 'robx':
                        robust_cf, _ = robx_algorithm(
                            X_train = X_train.to_numpy(),
                            predict_class_proba_fn = self.predict_fn_1,
                            start_counterfactual = cf_numpy,
                            tau = self.robXHparams['tau'],
                            variance = self.robXHparams['variance'],
                            N = self.robXHparams['N'],
                        )
                    case 'statrob':
                        robust_cf, artifacts_dict = statrobExplainer.optimize(
                            start_sample=cf_numpy.reshape(1, -1),
                            target_class=1 - orig_y,
                            method=self.config.get_config_by_key('statrobHparams')['method'],
                            desired_confidence=self.config.get_config_by_key('statrobHparams')['beta_confidence'],
                            opt_hparams=self.config.get_config_by_key('statrobHparams')['growingSpheresHparams']
                        )
                        self.results.add_record('artifacts', artifacts_dict)
                    case 'statrobxplus':
                        robust_cf = statrobxplusExplainer.optimize(
                            start_sample=cf_numpy.reshape(1, -1),
                            target_class=1 - orig_y,
                            desired_confidence=self.config.get_config_by_key('statrobHparams')['beta_confidence'],
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
                    cf = robust_cf.flatten(),
                    cf_desired_class=cf_desired_class,
                    x = x_numpy,
                    X_train=X_train.to_numpy(),
                    y_train=y_train,
                    lof_model = self.lof_model,
                    nearest_neighbors_model=self.nearest_neighbors,
                    predict_fn = self.predict_fn_1,
                    robXHparams = self.robXHparams
                )
            
                rob_metrics_2 = self.calculate_metrics(
                    cf = robust_cf.flatten(),
                    cf_desired_class=cf_desired_class,
                    x = x_numpy,
                    X_train=self.X_train2.to_numpy(),
                    y_train=self.y_train2,
                    lof_model = self.lof_model2,
                    nearest_neighbors_model=self.nearest_neighbors2,
                    predict_fn = self.predict_fn_2,
                    robXHparams = self.robXHparams
                )
                
            for k, v in rob_metrics.items():
                self.results.add_metric(f'robust_{k}', v)
                
            for k, v in rob_metrics_2.items():
                self.results.add_metric(f'robust_{k}_2', v)
            
            self.results.add_metric('robust_generation_time', rob_generation_time)
            
            self.results.add_record('base_cf', base_cf)
            self.results.add_record('robust_cf', robust_cf)
            self.results.add_record('orig_x', orig_x)
            self.results.add_record('orig_y', orig_y)
            
            # Calculate the L1&L2&CosineSim distance of the robust counterfactual to the base counterfactual
            if robust_cf is not None and base_cf is not None:
                self.results.add_metric('robust_cf_to_base_cf_proximity_L1', np.sum(np.abs(robust_cf - base_cf)))
                self.results.add_metric('robust_cf_to_base_cf_proximity_L2', np.sqrt(np.sum(np.square(robust_cf - base_cf))))
            else:
                self.results.add_metric('robust_cf_to_base_cf_proximity_L1', np.nan)
                self.results.add_metric('robust_cf_to_base_cf_proximity_L2', np.nan)
            
            if stop_after and j >= stop_after:
                print(f'Stopping after {stop_after} iterations.')
                break
            
            # Save the results to file after each iteration to avoid losing data
            if j % 30 == 0 or j == len(X_test) - 1 or j == stop_after - 1:
                results_dir = self.config.get_config_by_key('result_path')
                self.results.save_to_file(f'{results_dir}/{self.custom_experiment_name}.joblib')
            
        return True
    
    def calculate_metrics(self, cf: np.ndarray, 
                          cf_desired_class: int,
                          x: np.ndarray, 
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          lof_model: LocalOutlierFactor,
                          nearest_neighbors_model: NearestNeighbors,
                          predict_fn: callable,
                          robXHparams: dict
        ) -> dict:
        '''
        Calculates the metrics for a counterfactual example.
        
        Parameters:
            - cf: (np.ndarray) The counterfactual example.
            - cf_desired_class: (int) The desired class of the counterfactual example.
            - x: (np.ndarray) The original instance.
            - X_train: (np.ndarray) The training data.
            - y_train: (np.ndarray) The training labels.
            - lof_model: (object) The Local Outlier Factor model.
            - nearest_neighbors_model: (object) The Nearest Neighbors model.
            - predict_fn: (function) The function to be used for predictions.
            - robXHparams: (dict) The hyperparameters for the RobustX algorithm.
        
        Returns:
            - dict: The metrics.
        '''
        
        cf_label = predict_fn(cf)[0] > self.class_threshold
        
        # Validity
        validity = int(int(cf_label) == cf_desired_class)
        
        # Proximity L1
        proximityL1 = np.sum(np.abs(x - cf))
        
        # Proximity L2
        proximityL2 = np.sqrt(np.sum(np.square(x - cf)))
        
        # LOF
        lof = lof_model.score_samples(cf.reshape(1, -1))[0]
        
        # Counterfactual stability
        cf_counterfactual_stability = counterfactual_stability(
            cf, predict_fn, robXHparams['variance'], robXHparams['N']
        )
        
        # Discriminative Power (fraction of neighbors with the same label as the counterfactual)
        neigh_indices = nearest_neighbors_model.kneighbors(cf.reshape(1, -1), return_distance=False, n_neighbors=15)
        neigh_labels = y_train[neigh_indices[0]]
        dpow = np.sum(neigh_labels == cf_label) / len(neigh_labels) # The fraction of neighbors with the same label as the counterfactual
        
        # Plausibility (average distance to the 50 nearest neighbors in the training data)
        neigh_dist, _ = nearest_neighbors_model.kneighbors(cf.reshape(1, -1), return_distance=True, n_neighbors=50)
        plausibility = np.mean(neigh_dist[0])
        
        return {
            'validity': validity,
            'proximityL1': proximityL1,
            'proximityL2': proximityL2,
            'lof': lof,
            'cf_counterfactual_stability': cf_counterfactual_stability,
            'dpow': dpow,
            'plausibility': plausibility
        }
    
    def log_artifact(self, key: str, value: object) -> None:
        print(f'Accuracy of model 1: {self.acc1}')
        print(f'Accuracy of model 2: {self.acc2}')
           
    def get_results(self) -> ExperimentResults:
        return self.results
        
    
class TwoDatasetsExperiment(ExperimentBase):
    '''
    An experiment which trains two models on two samples given by ExperimentDataBase create method.
    '''
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
        
        self.preprocessor = self.experiment_data.preprocessor
        
        self.robXHparams = config.get_model_config('robXHparams')
        
        self.results = ExperimentResults()
        
        self.lof_model = LocalOutlierFactor(n_neighbors=50, novelty=True, p=1)
        self.lof_model.fit(self.X_train1.to_numpy()) # Fit the LOF model without column names
        self.lof_model2 = LocalOutlierFactor(n_neighbors=50, novelty=True, p=1)
        self.lof_model2.fit(self.X_train2.to_numpy())
        
        self.prep_done = False
    
    def prepare(self, 
            seed: int | tuple[int, int] = None,
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
        if self.model_type == 'mlp-sklearn':
            hparams = self.config.get_config_by_key('mlp-sklearn')
        elif self.model_type == 'rf-sklearn':
            hparams = self.config.get_config_by_key('rf-sklearn')
        elif self.model_type == 'mlp-torch':
            hparams = self.config.get_config_by_key('mlp-torch')
            
            if calibrate:
                raise ValueError('Calibration is not supported for PyTorch models.')
            
        else:
            raise ValueError('model_type must be either "mlp-sklearn" or "rf-sklearn"')

            
        if 'sklearn' in self.model_type:
            print('Training sklearn model')
            _model_type = self.model_type.split('-')[0].upper()
            
            if seed:
                if isinstance(seed, tuple):
                    hparams['random_state'] = seed[0]
                else:
                    hparams['random_state'] = seed
            
            if calibrate:
                self.model1 = train_calibrated_model(_model_type, self.X_train1, self.y_train1, 
                                                    base_model_hparams=hparams, calibration_method=calibrate_method)
            else:
                self.model1 = train_model(_model_type, self.X_train1, self.y_train1, hparams=hparams)
                
            if seed:
                if isinstance(seed, tuple):
                    hparams['random_state'] = seed[1]
                else:
                    hparams['random_state'] = seed
                
            self.model2 = train_model(_model_type, self.X_train2, self.y_train2, hparams=hparams)
            
            save_model(self.model1, f'models/{self.custom_experiment_name}_sample1.joblib')
            save_model(self.model2, f'models/{self.custom_experiment_name}_sample2.joblib')
            
            
            self.predict_fn_1 = scikit_predict_proba_fn(self.model1)
            self.predict_fn_1_crisp = scikit_predict_crisp_fn(self.model1)
            
            self.predict_fn_2 = scikit_predict_proba_fn(self.model2)
            self.predict_fn_2_crisp = scikit_predict_crisp_fn(self.model2)
           
            
        elif 'torch' in self.model_type:
            seed1 = self.config.get_config_by_key('random_state')
            seed2 = self.config.get_config_by_key('random_state')
            
            if seed:
                if isinstance(seed, tuple):
                    seed1 = seed[0]
                    seed2 = seed[1]
                else:
                    seed1 = seed
                    seed2 = seed
            
            
            print('Training torch model')
            self.model1 = MLPClassifier(
                input_dim=self.X_train1.shape[1],
                hidden_dims=hparams['hidden_dims'],
                activation=hparams['activation'],
                dropout=hparams['dropout'],
                seed=seed1
            )
                
            X_train1, X_val1, y_train1, y_val1 = train_test_split(self.X_train1, self.y_train1, test_size=0.15, random_state=42)
            self.model1.fit(
                X_train1, y_train1,
                X_val=X_val1, y_val=y_val1,
                epochs=hparams['epochs'],
                lr=hparams['lr'],
                batch_size=hparams['batch_size'],
                verbose=hparams['verbose'],
                early_stopping=hparams['early_stopping'],
            )
            self.predict_fn_1 = lambda x: self.model1.predict_proba(x)
            self.predict_fn_1_crisp = lambda x: self.model1.predict_crisp(x, threshold=self.class_threshold)
            
            self.model2 = MLPClassifier(
                input_dim=self.X_train2.shape[1],
                hidden_dims=hparams['hidden_dims'],
                activation=hparams['activation'],
                dropout=hparams['dropout'],
                seed=seed2
            )
            
            X_train2, X_val2, y_train2, y_val2 = train_test_split(self.X_train2, self.y_train2, test_size=0.15, random_state=42)
            self.model2.fit(
                X_train2, y_train2,
                X_val=X_val2, y_val=y_val2,
                epochs=hparams['epochs'],
                lr=hparams['lr'],
                batch_size=hparams['batch_size'],
                verbose=hparams['verbose'],
                early_stopping=hparams['early_stopping'],
            )
            
            self.predict_fn_2 = lambda x: self.model2.predict_proba(x)
            self.predict_fn_2_crisp = lambda x: self.model2.predict_crisp(x, threshold=self.class_threshold)
            
        else:
            raise ValueError('model_type unsupported')
            
        
        # Calculate the accuracy of the models
        self.acc1 = accuracy_score(self.y_test1, self.predict_fn_1_crisp(self.X_test1))
        self.acc2 = accuracy_score(self.y_test2, self.predict_fn_2_crisp(self.X_test2))
        
        self.log_artifact('accuracy_model1', self.acc1)
        self.log_artifact('accuracy_model2', self.acc2)
        
        self.prep_done = True
           

if __name__ == '__main__':
    config_wrapper = ConfigWrapper('config.yml')
    
    np.random.seed(config_wrapper.get_config_by_key('random_state'))
    results_dir = config_wrapper.get_config_by_key('result_path')
    
    experiments = [
        {
            'model_type': 'mlp-torch',
            'base_cf_method': 'gs',
            'calibrate': False,
            'calibrate_method': None,
            'custom_experiment_name': 'torch-fico-gs',
            'robust_method': 'statrob',
            'stop_after': None,
            'dataset': 'fico'
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
        
        dataset = Dataset(_exp['dataset'])
        
        e1 = SameSampleExperimentData(
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
            seed=(seed, seed + 1),
            calibrate=_exp['calibrate'],
            calibrate_method=_exp['calibrate_method']
        )
        
        run_status = td_exp.run(
            robust_method=_exp['robust_method'],
            base_cf_method=_exp['base_cf_method'],
            stop_after=_exp['stop_after']
        )
        
        results = td_exp.get_results()
        results.save_to_file(f'{results_dir}/{_exp["custom_experiment_name"]}.joblib')
        
    wandb_enabled = config_wrapper.get_config_by_key('wandb_enabled')
    if wandb_enabled:
        with open(config_wrapper.get_config_by_key('wandb_file'), 'r') as f:
            key = f.read()
        
        wandb.login(key=key)
        
        wandb.init(
            project=config_wrapper.get_config_by_key('wandb_project'),
            name=config_wrapper.get_config_by_key('experiment_name'),
        )
        # Log the config
        wandb.config.update(config_wrapper.get_entire_config())
        
        # Log experiment details list
        wandb.config.update({'experiments': experiments})
        
        
        
    
    import multiprocessing as mp
    
    
    for exp_config in experiments:
        
        processes = []
        for rep in range(1):
            
            p = mp.Process(target=__run_experiment, args=(exp_config, rep))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    print('All experiments finished.')
    
    
    if wandb_enabled:
        wandb.finish()