#!/bin/bash
#SBATCH --job-name=analysisPaper_$1
#SBATCH --output=analysisPaper_$1.out
#SBATCH --error=analysisPaper_$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=long
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=""

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=$P
source ~/.bashrc
conda activate mic
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
srun -u python -u tools/train.py ./configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --warp-cutmix True --bottom-pl-fill True --max-confidence True --seed 1 --deterministic --work-dir="./work_dirs/analysisPaper/$1$T" --auto-resume True --wandbid $1$T