#!/bin/bash
#SBATCH --job-name=baseline_synthia_mic_base_short_2gpu_resume
#SBATCH --output=baseline_synthia_mic_base_short_2gpu_resume.out
#SBATCH --error=baseline_synthia_mic_base_short_2gpu_resume.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=28271
source ~/.bashrc
conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x
srun -u python -u tools/train.py ./configs/mic/synthiaSeqHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=0 --l-mix-lambda=1.0 --work-dir="./work_dirs/synthia_baseline/synthia_mic_base_short_2gpu_resume02-27-21-30-00" --auto-resume True --wandbid synthia_mic_base_short_2gpu_resume02-27-21-30-00