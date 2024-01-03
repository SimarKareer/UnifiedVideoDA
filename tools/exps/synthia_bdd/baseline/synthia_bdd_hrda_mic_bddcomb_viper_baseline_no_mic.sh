#!/bin/bash
#SBATCH --job-name=analysisPaper_bddcomb_viper_baseline
#SBATCH --output=analysisPaper_bddcomb_viper_baseline.out
#SBATCH --error=analysisPaper_bddcomb_viper_baseline.err
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
srun python tools/train.py configs/mic/synthiaSeqHR2bddHR_mic_hrda_deeplab.py --launcher=slurm --no-masking True --l-warp-lambda=0.0 --l-mix-lambda=1.0 --seed 1 --deterministic --work-dir=./work_dirs/synthia_bdd/bddcomb_synthiaSeq_baseline_no_mic_11-01-14-13-00 --auto-resume True --wandbid bddcomb_synthiaSeq_baseline_no_mic_11-01-14-13-00