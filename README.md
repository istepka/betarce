## Counterfactual Explanations with Probabilistic Guarantees on their Robustness to Model Change


### Setup

#### 1. Setup Dependencies

Recursive clone: 

```bash
git clone --recurse-submodules https://github.com/istepka/CARLA.git
```

Or 

```bash
git clone https://github.com/istepka/CARLA.git 
git submodule update --init --recursive
```

#### 2. Install Environment

Conda: 
```bash
conda env create --name envname --file=environment.yml
```
or 
Pip (tested on Python 3.11.7): 
```bash
python -m pip install -r requirements.txt
```
***
### Usage

Directory structure:   
        `experiments`   
        ├── `notebooks`  
        ├── `visualizations`   
        `images`    
        `data`   
        `configs`    
        `src`    
        ├── `explainers`    
        ├── `datasets`  
        ├── `classifiers`  
        ├── `experiment.py`  
        `experiment_runner.py`  
  

*Make sure that your python is calling scripts from the root directory level to avoid issues with wrong paths  

***
### Reproduce experiments
To run experiments for the paper we utilized our internal compute cluster running on slurm. 
To facilitate efficient use of resources we created a script that runs experments on slurm via hydra framework.

Scripts for running experiments are located in `experiments` folder.

### Citation
If you find this work useful, please cite it as:
```
@inproceedings{stepka2025,
        author    = {Ignacy Stępka and Mateusz Lango and Jerzy Stefanowski},
        title     = {Counterfactual Explanations with Probabilistic Guarantees on their Robustness to Model Change},
        booktitle = {Proceedings of the 31st SIGKDD Conference on Knowledge Discovery and Data Mining},
        year      = {2025},
        month     = aug,
        address   = {Toronto, Canada},
      }
```