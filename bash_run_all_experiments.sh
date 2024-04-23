#!/bin/bash

# Define the directory containing the scripts
SCRIPT_DIRS=("bash_scripts/lgbm", "bash_scripts/v3")

# Export env variables
export USER_CONDA_ACT="[YOUR CONDA ACTIVATION PATH]"
export ENV_NAME="[YOUR CONDA ENVIRONMENT NAME]"

# Loop through each directory containing scripts
for dir in "${SCRIPT_DIRS[@]}"; do
    # Loop through each script in the directory
    for script in "$dir"/*.sh; do
        # Submit the script using bash
        bash "$script" # If you want to use slurm, use sbatch instead of bash
    done
done