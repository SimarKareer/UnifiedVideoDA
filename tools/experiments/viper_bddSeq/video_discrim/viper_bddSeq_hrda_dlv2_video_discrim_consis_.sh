#!/bin/bash
#SBATCH --job-name TEST_viper_bddSeq_hrda_video_disrim_pl_refinement_consis
#SBATCH --partition="hoffman-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=13
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="baymax,ig-88,brainiac,consu,spot,chappie,robby"
EXP_NAME=TEST_viper_bddSeq_hrda_video_disrim_pl_refinement_consis01-25-18-13-53
export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=37319
source ~/.bashrc
conda activate accel
cd /coc/scratch/vvijaykumar6/video_da_repo/mmseg
TMPDIR=/nethome/vvijaykumar6/w_logs
python ./tools/train.py configs/mic/viperHR2bddHR_mic_hrda_deeplab.py --launcher=slurm --no-masking True --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --adv-scale 1e-1 --warp-cutmix True --bottom-pl-fill True --consis-filter True --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME01-25-18-13-53 --auto-resume True --wandbid $EXP_NAME
