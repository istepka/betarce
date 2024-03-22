#!/bin/bash
#SBATCH --job-name=statrob    # Job name
#SBATCH -p dcc
#SBATCH -w dcc-3
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/serial_test_%j.log   # Standard output and error log

DATASET="wine"
EXPERIMENT="e1"
CONFIG_TO_USE="configv2_${DATASET}_${EXPERIMENT}.yaml"

conda activate robustcf

echo "Running experiment with config: ${CONFIG_TO_USE}"

python src/experimentsv2.py --config configs/${CONFIG_TO_USE} 

conda deactivate

echo "Job done"