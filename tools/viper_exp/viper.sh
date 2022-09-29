#!/bin/bash
#SBATCH --job-name=segformer-viper-$1
#SBATCH --output=logs$1.out
#SBATCH --error=logs$1.err
#SBATCH --gpus-per-task 1  # This gives 1 GPU to each process (or task)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2 # The number of processes for slurm to start on each node
#SBATCH --partition=short
#SBATCH --constraint="a40"

source ~/.bashrc

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

conda activate mmseg
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
srun python tools/train.py configs/segformer/segformer.b5.1024x1024.viper.160k.py --launcher="slurm"