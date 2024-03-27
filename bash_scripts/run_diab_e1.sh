#!/bin/bash
#SBATCH -p pmem
#SBATCH -w pmem-1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time=144:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

DATASET="diabetes"
EXPERIMENT="e1"
CONFIG_TO_USE="configs/configv2_${DATASET}_${EXPERIMENT}.yml"

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiment with config: ${CONFIG_TO_USE}"

python ./src/experimentsv2.py --config $CONFIG_TO_USE 


echo "Job done"
