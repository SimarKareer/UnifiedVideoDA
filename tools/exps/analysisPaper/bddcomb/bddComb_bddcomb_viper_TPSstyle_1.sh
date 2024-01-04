#!/bin/bash
#SBATCH --job-name=analysisPaper_bddcomb_viper_TPSstyle
#SBATCH --output=analysisPaper_bddcomb_viper_TPSstyle.out
#SBATCH --error=analysisPaper_bddcomb_viper_TPSstyle.err
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
export MASTER_PORT=32426
source ~/.bashrc
conda activate micExp
cd ~/flash9/oldFlash/VideoDA/experiments/mmsegmentationExps

set -x
srun python tools/train.py configs/mic/viperHR2bddHR_mic_hrda_deeplab.py --launcher="slurm" --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --warp-cutmix True --bottom-pl-fill True --TPS-warp-pl-confidence True --no-masking True --seed 1 --deterministic --work-dir=./work_dirs/viper_bdd/bddcomb_viper_TPSstyle11-02-18-47-34 --auto-resume True --wandbid bddcomb_viper_TPSstyle11-02-18-47-34