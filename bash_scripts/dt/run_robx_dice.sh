#!/bin/bash
#SBATCH -p obl
#SBATCH -w obl2
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time=144:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

CONFIG_TO_USE="configs/dt/configv2_robx_dice.yml"

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiment with config: ${CONFIG_TO_USE}"

python ./src/experimentsv2.py --config $CONFIG_TO_USE 


echo "Job done"
