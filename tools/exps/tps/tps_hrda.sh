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
srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=1 --l-mix-lambda=0 --l-warp-begin=1500 --bottom-pl-fill True --no-masking True --TPS-warp-pl-confidence True --TPS-warp-pl-confidence-thresh 0.0 --lr 6e-5 --total-iters=40000 --seed 1 --deterministic --work-dir="./work_dirs/viper_cs/tps_exp/$1$T" --wandbid $1$T
