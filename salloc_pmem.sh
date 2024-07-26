#!/bin/bash
#SBATCH --job-name=interactive
#SBATCH --output=slurm_logs/interactive_%A_%a.out
#SBATCH --error=slurm_logs/interactive_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=pmem
#SBATCH --nodelist=pmem-4

watch ls 
