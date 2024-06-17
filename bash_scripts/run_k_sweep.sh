#!/bin/bash
#SBATCH -p pmem
#SBATCH -w pmem-6
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time=144:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/k_sweep_%j.out
#SBATCH --error=logs/k_sweep_%j.err


CONFIG_TO_USE="configs/configv2_k_sweep.yml"

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiment with config: ${CONFIG_TO_USE}"

python ./src/experimentsv2.py --config $CONFIG_TO_USE 


echo "Job done"