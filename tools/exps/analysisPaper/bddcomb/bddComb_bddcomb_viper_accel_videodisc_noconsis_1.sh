#!/bin/bash
#SBATCH --job-name=analysisPaper_bddcomb_viper_accel_videodisc_noconsis
#SBATCH --output=analysisPaper_bddcomb_viper_accel_videodisc_noconsis.out
#SBATCH --error=analysisPaper_bddcomb_viper_accel_videodisc_noconsis.err
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
export MASTER_PORT=34380
source ~/.bashrc
conda activate micExp
cd ~/flash9/oldFlash/VideoDA/experiments/mmsegmentationExps

set -x
srun python tools/train.py configs/mic/viperHR2bddHR_mic_hrda_deeplab.py --launcher="slurm" --l-mix-lambda=1.0 --l-warp-lambda=-1.0  --no-masking True --adv-scale 1e-1 --seed 1 --deterministic --work-dir=./work_dirs/viper_bdd/bddcomb_viper_accel_videodisc_noconsis11-02-14-51-36 --auto-resume True --wandbid bddcomb_viper_accel_videodisc_noconsis11-02-14-51-36