#!/bin/bash
#SBATCH --job-name=lwarpv3_$1
#SBATCH --output=lwarpv3_$1.out
#SBATCH --error=lwarpv3_$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=$P
source ~/.bashrc
conda activate mic3
cd ~/flash/Projects/VideoDA/experiments/mmsegmentationExps

set -x
srun -u python -u tools/train.py ./configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=0.1 --l-mix-lambda=1.0 --work-dir="./work_dirs/lwarpv3/$1$T" --auto-resume True --wandbid $1$T
