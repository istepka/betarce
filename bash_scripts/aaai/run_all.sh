#!/bin/bash
#SBATCH --job-name=RCE_all
#SBATCH --output=slurm_logs/0208_all_betarce/l_%A_%a.out
#SBATCH --error=slurm_logs/0208_all_betarce/l_%A_%a.err
#SBATCH --array=0-35
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --partition=obl
#SBATCH --nodelist=obl2
#SBATCH --distribution=pack


# Array of configuration files
configs=(
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_bre_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_bre_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_bre_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_bre_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_bre_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_bre_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_bre_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_bre_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_bre_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_win_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_win_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_win_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_win_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_win_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_win_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_win_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_win_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_win_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_dia_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_dia_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_dia_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_dia_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_dia_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_dia_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_dia_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_dia_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_dia_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_fic_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_fic_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_fic_Arc.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_fic_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_fic_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_fic_Boo.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/gs_bet_fic_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/dic_bet_fic_See.yml"
"./bash_scripts/aaai/all/all_betarce_02_08/fac_bet_fic_See.yml"
)

# Get the configuration file for this job array task
config=${configs[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected configuration
source /home/inf148179/anaconda3/bin/activate
conda activate betarce
echo "Running experiments"
python ./src/experimentsv3.py --config $config