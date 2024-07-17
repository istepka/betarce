#!/bin/bash
#SBATCH --job-name=betaRCE
#SBATCH --output=slurm_logs/aaai2/nn_exp_%A_%a.out
#SBATCH --error=slurm_logs/aaai2/nn_exp_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --partition=obl
#SBATCH --nodelist=obl2

# Array of configuration files
configs=(
    "./bash_scripts/aaai/dice/nn_breast.yml"
    "./bash_scripts/aaai/dice/nn_wine.yml"
    "./bash_scripts/aaai/dice/nn_fico.yml"
    "./bash_scripts/aaai/dice/nn_diabetes.yml"
    "./bash_scripts/aaai/face/nn_breast.yml"
    "./bash_scripts/aaai/face/nn_wine.yml"
    "./bash_scripts/aaai/face/nn_fico.yml"
    "./bash_scripts/aaai/face/nn_diabetes.yml"
    "./bash_scripts/aaai/gs/nn_breast.yml"
    "./bash_scripts/aaai/gs/nn_wine.yml"
    "./bash_scripts/aaai/gs/nn_fico.yml"
    "./bash_scripts/aaai/gs/nn_diabetes.yml"
    "./bash_scripts/aaai/other/face.yml"
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
python ./src/experimentsv3.py --config $config