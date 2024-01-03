#!/bin/bash
#SBATCH --job-name=analysisPaper_bdd_source
#SBATCH --output=analysisPaper_bdd_source.out
#SBATCH --error=analysisPaper_bdd_source.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude="baymax"

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=37319
source ~/.bashrc
conda activate mic
cd ~/flash9/oldFlash/VideoDA/mmsegmentation

set -x
srun python tools/train.py configs/mic/bddHR_mic_hrda_deeplab_supervised.py --launcher=slurm --no-masking True --source-only2 True --l-warp-lambda=0.0 --l-mix-lambda=1.0 --seed 1 --deterministic --work-dir=./work_dirs/synthia_bdd/bdd_dg_supervised_11-02-21-26-00 --auto-resume True --wandbid bdd_dg_supervised_11-02-21-26-00