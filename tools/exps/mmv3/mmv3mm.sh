#!/bin/bash
#SBATCH --job-name=mmv3mm$1
#SBATCH --output=mmv3mm$1.out
#SBATCH --error=mmv3mm$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude="optimistprime"

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=$P
source ~/.bashrc
conda activate mic
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
srun -u python -u ./tools/train.py ./configs/mic/viperHR2csHR_mic_daformer_mm.py --launcher="slurm" --l-warp-lambda -1 --l-mix-lambda 1 --imnet-feature-dist-lambda=0 --optimizer adamw --lr-schedule poly_10_warm --lr 6e-5 --total-iters=40000 --work-dir="./work_dirs/mmv3/$1$T" --wandbid $1$T

