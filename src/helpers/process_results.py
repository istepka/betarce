import pandas as pd 
import numpy as np
from tqdm import tqdm
import os 
 

def process_results(results_path: str, ext: str = 'feather'):
    results_list = []

    # Read all the results from results/ directory
    for dirname in tqdm(os.listdir(results_path), desc='Reading results'):
        if 'images' not in dirname:
            if not os.path.isdir(f'{results_path}/{dirname}') or 'results' not in os.listdir(f'{results_path}/{dirname}'):
                continue
            for file in os.listdir(f'{results_path}/{dirname}/results'):
                if file.endswith(f'.{ext}'):
                    df = pd.read_feather(f'{results_path}/{dirname}/results/{file}')
                    
                    # Custom parsing kinda weird
                    if 'dt' in dirname:
                        df['base_model'] = 'DecisionTree'
                    else:
                        df['base_model'] = 'NeuralNetwork'
                        
                    if 'dice' in dirname:
                        df['base_cf_method'] = 'Dice'
                    else:
                        df['base_cf_method'] = 'GrowingSpheres'
                        
                    # Temporary fix for the experiment type
                    # ----------------------------
                    # exp_types = list(filter(lambda x: 'seed' not in x.lower(), df['experiment_type'].unique()))
                    # df = df[df['experiment_type'].isin(exp_types)]
                    # ----------------------------
                        
                    results_list.append(df)
    
    # Concatenate all the results
    raw_df: pd.DataFrame = pd.concat(results_list, ignore_index=True)
    return raw_df

if __name__ == '__main__':
    
    # Make sure we are in the right directory
    while 'src' in os.getcwd():
        os.chdir('..')
        
    print(f'Current directory: {os.getcwd()}')
    
    # Process the results
    paths = {
        'results_lgbm': ['main', 'robx'],
        'results_v3': ['main', 'robx', 'confidence', 'k', 'generalization']
    }
    
    for path, subdirs in paths.items():
        for subdir in subdirs:
            if subdir == 'main':
                df = process_results(f'{path}')
            else:
                df = process_results(f'{path}/{subdir}')
            df.to_feather(f'{path}/{subdir}_results.feather')