#!/bin/bash

#SBATCH --job-name TESTING_viper_bddSeq_hrda_dlv2_accel_consis_mixup_video_discrim_consis_filter
#SBATCH --partition="hoffman-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=13
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="baymax, heistotron,ig-88,megabot"

EXP_NAME=TESTING_viper_bddSeq_hrda_dlv2_accel_consis_mixup_video_discrim_consis_filter_01-24-00-18-30

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=37319
source ~/.bashrc
conda activate accel
cd /coc/scratch/vvijaykumar6/video_da_repo/mmseg


set -x
python ./tools/train.py configs/mic/viperHR2bddHR_mic_hrda_deeplab.py --no-masking True --launcher=slurm --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --adv-scale 1e-1 --warp-cutmix True --bottom-pl-fill True --consis-filter True --accel True --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME --auto-resume True --wandbid $EXP_NAME