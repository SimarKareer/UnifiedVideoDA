#!/bin/bash
#SBATCH --job-name=cache$1
#SBATCH --output=cache$1.out
#SBATCH --error=cache$1.err
#SBATCH --gpus-per-task 1  # This gives 1 GPU to each process (or task)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 # The number of processes for slurm to start on each node
#SBATCH --partition=short
#SBATCH --constraint="a40"
export PYTHONUNBUFFERED=TRUE

source ~/.bashrc

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

conda activate mmseg
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x

python tools/test.py ./configs/segformer/segformer.b5.1024x1024.viper.160k.py ./work_dirs/segformer.b5.1024x1024.viper.160k/iter_48000.pth --cache work_dirs/sourceModelCacheLogits --launcher none
