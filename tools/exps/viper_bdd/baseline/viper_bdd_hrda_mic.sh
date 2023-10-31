#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=logs_$1.out
#SBATCH --error=logs_$1.err
#SBATCH --gpus-per-task 1  # This gives 1 GPU to each process (or task)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4 # The number of processes for slurm to start on each node
#SBATCH --partition=short
#SBATCH --constraint="a40"

source ~/.bashrc

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x
srun python tools/train.py configs/mic/viperHR2bddHR_mic_hrda.py --launcher=slurm --l-warp-lambda=0.0 --l-mix-lambda=1.0 --seed 1 --deterministic --work-dir=./work_dirs/viper_bdd/$1$T --auto-resume True --nowandb True