#!/bin/bash
#SBATCH --job-name=betaRCE
#SBATCH --output=slurm_logs/1807/nn_exp_%A_%a.out
#SBATCH --error=slurm_logs/1807/nn_exp_%A_%a.err
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=13G
#SBATCH --time=168:00:00
#SBATCH --partition=obl
#SBATCH --nodelist=obl2

# Array of configuration files
configs=(
    # dice configurations
    # "./bash_scripts/aaai/dice/nn_breastA.yml"
    # "./bash_scripts/aaai/dice/nn_breastB.yml"
    # "./bash_scripts/aaai/dice/nn_breastS.yml"
    # "./bash_scripts/aaai/dice/nn_diabetesA.yml"
    # "./bash_scripts/aaai/dice/nn_diabetesB.yml"
    # "./bash_scripts/aaai/dice/nn_diabetesS.yml"
    # "./bash_scripts/aaai/dice/nn_ficoA.yml"
    # "./bash_scripts/aaai/dice/nn_ficoB.yml"
    # "./bash_scripts/aaai/dice/nn_ficoS.yml"
    # "./bash_scripts/aaai/dice/nn_wineA.yml"
    # "./bash_scripts/aaai/dice/nn_wineB.yml"
    # "./bash_scripts/aaai/dice/nn_wineS.yml"
    
    # face configurations
    "./bash_scripts/aaai/face/nn_breastA.yml"
    "./bash_scripts/aaai/face/nn_breastB.yml"
    "./bash_scripts/aaai/face/nn_breastS.yml"
    "./bash_scripts/aaai/face/nn_diabetesA.yml"
    "./bash_scripts/aaai/face/nn_diabetesB.yml"
    "./bash_scripts/aaai/face/nn_diabetesS.yml"
    "./bash_scripts/aaai/face/nn_ficoA.yml"
    "./bash_scripts/aaai/face/nn_ficoB.yml"
    "./bash_scripts/aaai/face/nn_ficoS.yml"
    "./bash_scripts/aaai/face/nn_wineA.yml"
    "./bash_scripts/aaai/face/nn_wineB.yml"
    "./bash_scripts/aaai/face/nn_wineS.yml"
    
    # gs configurations
    # "./bash_scripts/aaai/gs/nn_breastA.yml"
    # "./bash_scripts/aaai/gs/nn_breastB.yml"
    # "./bash_scripts/aaai/gs/nn_breastS.yml"
    # "./bash_scripts/aaai/gs/nn_diabetesA.yml"
    # "./bash_scripts/aaai/gs/nn_diabetesB.yml"
    # "./bash_scripts/aaai/gs/nn_diabetesS.yml"
    # "./bash_scripts/aaai/gs/nn_ficoA.yml"
    # "./bash_scripts/aaai/gs/nn_ficoB.yml"
    # "./bash_scripts/aaai/gs/nn_ficoS.yml"
    # "./bash_scripts/aaai/gs/nn_wineA.yml"
    # "./bash_scripts/aaai/gs/nn_wineB.yml"
    # "./bash_scripts/aaai/gs/nn_wineS.yml"
)

# Get the configuration file for this job array task
config=${configs[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected configuration
source /home/inf148179/anaconda3/bin/activate
conda activate betarce
echo "Running experiments"
python ./src/experimentsv3.py --config $config