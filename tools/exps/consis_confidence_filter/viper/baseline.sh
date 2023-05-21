#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=$1.out
#SBATCH --error=$1.err
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

#change begin to 1500
# LOAD_FROM="work_dirs/consis_confidence_filter_train/mic_rgb_dacs_cleaned_consis_conf_filter_corr05-13-16-53-54/iter_12000.pth"
srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=-1 --l-mix-lambda=1 --lr 24e-5 --total-iters=10000 --seed 74111605 --deterministic --work-dir="./work_dirs/viper_cs/4gpu_baselines/lr_sweep/$1$T" --wandbid $1$T
# --wandbid $1$T