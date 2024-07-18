#!/bin/bash
#SBATCH --job-name=BRCE_base
#SBATCH --output=slurm_logs/1807/nn_exp_%A_%a.out
#SBATCH --error=slurm_logs/1807/nn_exp_%A_%a.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --partition=obl

# Array of configuration files
configs=(
    "./bash_scripts/aaai/other/face.yml"
    "./bash_scripts/aaai/other/dice.yml"
    "./bash_scripts/aaai/other/rbr.yml"
    "./bash_scripts/aaai/other/roar.yml"
    "./bash_scripts/aaai/other/gs.yml"
)

# Get the configuration file for this job array task
config=${configs[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected configuration
source /home/inf148179/anaconda3/bin/activate
conda activate betarce
echo "Running experiments"
python ./src/experimentsv3_justbase.py --config $config