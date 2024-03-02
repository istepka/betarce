from config_wrapper import ConfigWrapper
from experiments_helpers import SameSampleExperimentData, TwoDatasetsExperiment, TwoSamplesOneDatasetExperimentData
from create_data_examples import Dataset
from copy import deepcopy
import numpy as np
import wandb
import argparse
import random

parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--config', type=str, help='Path to the config file')
# Flag to enable wandb logging
parser.add_argument('--wandb', type=bool, help='Enable wandb logging', default=False)
# Experiment name
parser.add_argument('--experiment', type=str, help='Name of the experiment', default='default')
parser.add_argument('--base_cf_method', type=str, help='Name of the base counterfactual method', default='gs')
parser.add_argument('--model_type', type=str, help='Type of the model', default='mlp-torch')
parser.add_argument('--robust_method', type=str, help='Robustness method', default='statrob')
parser.add_argument('--stop_after', type=int, help='Stop after n iterations', default=None)
parser.add_argument('--dataset', type=str, help='Name of the dataset', default='fico')
parser.add_argument('--experiment_type', type=str, help='Name of the experiment data class', default='SameSampleExperimentData')

args = parser.parse_args()

config_wrapper = ConfigWrapper('config.yml' if args.config is None else args.config)
    
np.random.seed(config_wrapper.get_config_by_key('random_state'))
random.seed(config_wrapper.get_config_by_key('random_state'))
results_dir = config_wrapper.get_config_by_key('result_path')

experiments = [
    {
        'model_type': 'mlp-torch' if args.model_type is None else args.model_type,
        'base_cf_method': 'gs' if args.base_cf_method is None else args.base_cf_method,
        'calibrate': False,
        'calibrate_method': None,
        'custom_experiment_name': 'torch-fico-gs' if args.experiment is None else args.experiment,
        'robust_method': 'statrob' if args.robust_method is None else args.robust_method,
        'stop_after': None if args.stop_after is None else args.stop_after,
        'dataset': 'fico'  if args.dataset is None else args.dataset,
        'experiment_type': 'SameSampleExperimentData' if args.experiment_type is None else args.experiment_type,
    }
]

print(experiments)
print(config_wrapper.get_entire_config())


def __run_experiment(exp_config: dict, rep: int):
    _exp = deepcopy(exp_config)
    _exp['custom_experiment_name'] = f"{_exp['custom_experiment_name']}_{rep}"
    
    # Set the random state for reproducibility, by adding the rep number to the seed
    _config_wrapper = config_wrapper.copy()
    seed = _config_wrapper.get_config_by_key('random_state') + rep
    seed = int(seed)
    _config_wrapper.set_config_by_key('random_state', seed)
    
    dataset = Dataset(_exp['dataset'])
    
    if _exp['experiment_type'] == 'SameSampleExperimentData':
        e1 = SameSampleExperimentData(
            dataset, 
            random_state=seed,
            one_hot_encode=True,
            standardize='minmax'
        )
    else:
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
    
wandb_enabled = config_wrapper.get_config_by_key('wandb_enabled') if args.wandb is None else args.wandb
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
    
    
    

# import multiprocessing as mp


for exp_config in experiments:
    
    for rep in range(1):
        __run_experiment(exp_config, rep)
        
        
print('All experiments finished.')


if wandb_enabled:
    wandb.finish()