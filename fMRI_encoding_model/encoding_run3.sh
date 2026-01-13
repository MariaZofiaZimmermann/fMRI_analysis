#!/bin/bash

#SBATCH --job-name="encodingPJM"
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --account=g100-2194
#SBATCH --partition=okeanos
#SBATCH --output=logs/encoding_%A_%a.out
#SBATCH --error=logs/encoding_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --array=1-22
#SBATCH --mail-user=your_email@domain.tld
#SBATCH --mail-type=ALL

# --- Diagnostics ---
set -x
echo "STARTING JOB: $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# --- Load Anaconda and source conda.sh manually ---

source /lustre/tetyda/apps/python/anaconda3.8/etc/profile.d/conda.sh
conda activate /lustre/tetyda/home/perpetua/.conda/envs/vena39
# --- Set environment variable for data path ---
export STUDY=/home/perpetua

# --- Run the Python script ---
python Hridge3_PJM.py

