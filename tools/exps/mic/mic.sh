#!/bin/bash
#SBATCH --job-name=mic$1
#SBATCH --output=mic$1.out
#SBATCH --error=mic$1.err
#SBATCH --gpus-per-task 1  # This gives 1 GPU to each process (or task)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 # The number of processes for slurm to start on each node
#SBATCH --partition=short
#SBATCH --constraint="a40"
export PYTHONUNBUFFERED=TRUE

source ~/.bashrc

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

conda activate mic
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x

python run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda.py
