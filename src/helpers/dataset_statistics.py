import pandas as pd 
import os 

print(os.getcwd())

from data_handler import Dataset

datasets = ['fico', 'wine_quality', 'diabetes', 'breast_cancer']
for dataset in datasets:
    d = Dataset(dataset)
    print(f'\n\nDataset: {dataset}')
    rows, cols = d.get_raw_df().shape
    print(f'Rows: {rows}, Cols: {cols}')

    target_name = d.get_target_column()
    imbalance = d.get_raw_df()[target_name].value_counts(normalize=True)
    ir = max(imbalance) / min(imbalance)
    print(f'Imbalance ratio: {ir:.2f}')