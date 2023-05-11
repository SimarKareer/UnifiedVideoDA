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

# LOAD_FROM="/coc/testnvme/skareer6/Projects/VideoDA/experiments/mmsegmentationExps/work_dirs/baselines/hrda-rgb-sourceonly04-29-13-14-01/iter_15000.pth"
# LOAD_FROM="/coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/lwarp/1gbaseline/iter_40000.pth"
LOAD_FROM="/coc/scratch/vvijaykumar6/mmseg/work_dirs/consis_confidence_filter_train/mic_resume_consis_conf_filter_0_99_4gpu05-09-18-28-35/latest.pth"
srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --optimizer adamw --lr-schedule poly_10_warm --lr 6e-5 --total-iters=1 --eval csseq --work-dir="./work_dirs/consis_confidence_filter/mic_resume_consis_conf_filter_0_99_4gpu05-09-18-28-35_eval" --wandbid mic_resume_consis_conf_filter_0_99_4gpu05-09-18-28-35_eval --load-from $LOAD_FROM