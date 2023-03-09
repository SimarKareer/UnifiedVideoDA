#!/bin/bash
#SBATCH --job-name=baseline_$1
#SBATCH --output=baseline_$1.out
#SBATCH --error=baseline_$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=$P
source ~/.bashrc
conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x
srun -u python -u tools/train.py ./configs/mic/synthiaSeqHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=0 --l-mix-lambda=1.0 --work-dir="./work_dirs/synthia_baseline/$1$T" --auto-resume True --wandbid $1$T