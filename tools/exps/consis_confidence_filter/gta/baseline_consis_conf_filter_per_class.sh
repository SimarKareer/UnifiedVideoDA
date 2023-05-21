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
srun -u python -u ./tools/train.py configs/mic/gtaHR2csHR_mic_hrda_VDA.py --launcher="slurm" --l-warp-lambda=1 --l-warp-begin=1500 --l-mix-lambda=1 --bottom-pl-fill True --consis-confidence-filter True --consis-confidence-thresh 0.95 --consis-confidence-per-class-thresh True --lr 27e-5 --total-iters=10000 --seed 1 --deterministic --work-dir="./work_dirs/gta_cs/consis_confidence_filter_train/$1$T" --wandbid $1$T
# --wandbid $1$T