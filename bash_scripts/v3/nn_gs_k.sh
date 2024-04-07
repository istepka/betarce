#!/bin/bash
#SBATCH -p pmem       # Use the appropriate partition
#SBATCH -w pmem-1
#SBATCH --nodes 1         # Request 1 node
#SBATCH --ntasks-per-node 1  
#SBATCH --cpus-per-task 1     
#SBATCH --time=144:00:00       
#SBATCH --output=logs_v3/output_%j.out
#SBATCH --error=logs_v3/error_%j.err

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiments"
python ./src/experimentsv3.py --config configs/v3/gs_nn_k.yml 

echo "Job done"
