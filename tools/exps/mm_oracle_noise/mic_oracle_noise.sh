#!/bin/bash
#SBATCH --job-name=mic_baseline_$1
#SBATCH --output=mic_baseline_$1.out
#SBATCH --error=mic_baseline_$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude="ig-88,perseverance,cheetah,claptrap"

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=$P
source ~/.bashrc
conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x

srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-mix-lambda 1.0 --l-warp-lambda -1.0 --oracle-mask-add-noise True --oracle-mask-noise-percent 0.1 --optimizer adamw --lr-schedule poly_10_warm --lr 24e-5 --total-iters=15000 --work-dir="./work_dirs/baselines/$1$T" --wandbid $1$T