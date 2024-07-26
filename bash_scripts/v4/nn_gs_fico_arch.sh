#!/bin/bash
#SBATCH -p hgx       # Use the appropriate partition
#SBATCH -w hgx2
#SBATCH --nodes 1         # Request 1 node
#SBATCH --ntasks-per-node 1  
#SBATCH --cpus-per-task 1     
#SBATCH --time=144:00:00       
#SBATCH --output=logs_v4/output_%j.out
#SBATCH --error=logs_v4/error_%j.err

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiments"
python ./src/experimentsv3.py --config configs/v4/gs_nn_fico_arch.yml 

echo "Job done"
