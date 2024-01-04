#!/bin/bash
#SBATCH --job-name=analysisPaper_bddcomb_synthia_DAVSNstyle
#SBATCH --output=analysisPaper_bddcomb_synthia_DAVSNstyle.out
#SBATCH --error=analysisPaper_bddcomb_synthia_DAVSNstyle.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=long
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude="baymax"

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=31285
source ~/.bashrc
conda activate accel
cd /coc/scratch/vvijaykumar6/video_da_repo/mmseg

set -x
srun python tools/train.py configs/mic/synthiaHR2bddHR_mic_hrda_deeplab.py --launcher="slurm" --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --warp-cutmix True --bottom-pl-fill True --max-confidence True --adv-scale 1e-1 --no-masking True --seed 1 --deterministic --work-dir=./work_dirs/synthia_bdd/bddcomb_synthia_DAVSNstyle11-02-18-42-41 --auto-resume True --wandbid bddcomb_synthia_DAVSNstyle11-02-18-42-41