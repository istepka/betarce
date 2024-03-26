#!/bin/bash
#SBATCH --partition hgx
#SBATCH -w hgx2
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time=144:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/generalization_%j.out
#SBATCH --error=logs/generalization_%j.err


CONFIG_TO_USE="configs/configv2_generalization.yml"

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiment with config: ${CONFIG_TO_USE}"

python ./src/experimentsv2.py --config $CONFIG_TO_USE 


echo "Job done"
