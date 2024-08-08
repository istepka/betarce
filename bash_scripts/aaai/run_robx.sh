#!/bin/bash
#SBATCH --job-name=RCE_Robx
#SBATCH --output=slurm_logs/2907_robx/l_%A_%a.out
#SBATCH --error=slurm_logs/2907_robx/l_%A_%a.err
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=168:00:00
#SBATCH --partition=obl
#SBATCH --nodelist=obl2

# Array of configuration files
configs=(
"./bash_scripts/aaai/robx/robx/gs_rob_bre_Arc.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_bre_Arc.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_bre_Arc.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_bre_Boo.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_bre_Boo.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_bre_Boo.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_bre_See.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_bre_See.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_bre_See.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_win_Arc.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_win_Arc.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_win_Arc.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_win_Boo.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_win_Boo.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_win_Boo.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_win_See.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_win_See.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_win_See.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_dia_Arc.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_dia_Arc.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_dia_Arc.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_dia_Boo.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_dia_Boo.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_dia_Boo.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_dia_See.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_dia_See.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_dia_See.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_fic_Arc.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_fic_Arc.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_fic_Arc.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_fic_Boo.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_fic_Boo.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_fic_Boo.yml"
"./bash_scripts/aaai/robx/robx/gs_rob_fic_See.yml"
"./bash_scripts/aaai/robx/robx/dic_rob_fic_See.yml"
"./bash_scripts/aaai/robx/robx/fac_rob_fic_See.yml"
)

# Get the configuration file for this job array task
config=${configs[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected configuration
source /home/inf148179/anaconda3/bin/activate
conda activate betarce
echo "Running experiments"
python ./src/experimentsv3.py --config $config