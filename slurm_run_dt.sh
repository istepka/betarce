#!/bin/bash

# Define the directory containing the scripts
SCRIPT_DIR="bash_scripts/dt"

# Loop through each script in the directory
for script in "$SCRIPT_DIR"/*.sh; do
    # Submit the script using sbatch
    sbatch "$script"
done
