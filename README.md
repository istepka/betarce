## Counterfactual Explanations Robust to Blackbox Model Change through Beta distribution based statistical robustness measure


### Installation
1. Prepare a python environment
    ```bash
    pip install -r requirements.txt
    ```
2. (optional) If you want to run experiments with wandb logging enabled, you need to create a `wandb.key` file with your wandb key in it. 
    ```bash
    echo "your_wandb_key" > wandb.key
    ```

### Usage
- Generally speaking, the most important file is `config.yml` which contains all the parameters used by the code running experiments.
- Running experiments from console.
    Example 1:  
    ```bash
    python src/run_experiments.py --dataset fico --experiment torch-fico-gs-a9 --stop_after 20 --config config.yml
    ```
- Results analysis can be performed using `src/results_analysis2.ipynb` notebook.

