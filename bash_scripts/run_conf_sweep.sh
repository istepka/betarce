#!/bin/bash
#SBATCH --job-name=conf_sweep    # Job name
#SBATCH --partition obl
#SBATCH -w obl1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/conf_sweep_%j.out
#SBATCH --error=logs/conf_sweep_%j.err


CONFIG_TO_USE="configs/configv2_confidence_sweep.yml"

source /home/inf148179/anaconda3/bin/activate
conda activate robustcf

echo "Running experiment with config: ${CONFIG_TO_USE}"

python ./src/experimentsv2.py --config $CONFIG_TO_USE 


echo "Job done"
