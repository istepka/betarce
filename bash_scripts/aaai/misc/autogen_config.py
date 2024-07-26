import yaml
import time
import os
from copy import deepcopy

SOURCE_YML = 'bash_scripts/aaai/misc/template_K_sweep.yml'
CUSTOM_NAME = 'betarob_k_sweep'


SOURCE_YML_DIR = os.path.dirname(SOURCE_YML)

with open(SOURCE_YML) as f:
    d = yaml.load(f, Loader=yaml.FullLoader)

RESULTS_PATH = f'/home/inf148179/robust-cf/aaai_experiments/results_aaai/{CUSTOM_NAME}/'
LOGS_PATH = f'/home/inf148179/robust-cf/aaai_experiments/logs_aaai/{CUSTOM_NAME}/'

month_day = time.strftime("%m%d")


for dataset in ['breast_cancer', 'wine_quality', 'diabetes', 'fico']:
    for ex_type in ['Architecture', 'Bootstrap', 'Seed']:
        for base_method in ['gs', 'dice']:
        
            tmp_d = deepcopy(d)
            robust_method = tmp_d['experiments_setup']['robust_cf_method']
            
            tmp_d['experiments_setup']['ex_types'] = [ex_type]
            tmp_d['experiments_setup']['datasets'] = [dataset]
            tmp_d['experiments_setup']['base_counterfactual_method'] = base_method
            
            tmp_d['general']['result_path'] = RESULTS_PATH + f'{month_day}/{dataset}/{ex_type}/{base_method}/{robust_method}/'
            tmp_d['general']['log_path'] = LOGS_PATH + f'{month_day}/{dataset}/{ex_type}/{base_method}/{robust_method}/'
            
            if not os.path.exists(tmp_d['general']['result_path']):
                os.makedirs(tmp_d['general']['result_path'])
                
            if not os.path.exists(tmp_d['general']['log_path']):
                os.makedirs(tmp_d['general']['log_path'])
                
            if not os.path.exists(f'{SOURCE_YML_DIR}/{CUSTOM_NAME}'):
                os.makedirs(f'{SOURCE_YML_DIR}/{CUSTOM_NAME}')
                
            filename = f'{SOURCE_YML_DIR}/{CUSTOM_NAME}/{base_method[:3]}_{robust_method[:3]}_{dataset[:3]}_{ex_type[:3]}.yml'
                
            with open(filename, 'w') as f:
                yaml.dump(tmp_d, f)
                
            # Verify if the file was created
            if not os.path.exists(filename):
                print(f'Error creating file \"{filename}\"')
            
                
            print(f'\"./{filename}\"')
        
print('Done')
        
        