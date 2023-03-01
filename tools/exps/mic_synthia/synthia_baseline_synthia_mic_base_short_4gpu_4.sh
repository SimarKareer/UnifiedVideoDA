#!/bin/bash
#SBATCH --job-name=baseline_synthia_mic_base_short_4gpu
#SBATCH --output=baseline_synthia_mic_base_short_4gpu.out
#SBATCH --error=baseline_synthia_mic_base_short_4gpu.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=29711
source ~/.bashrc
conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x
srun -u python -u tools/train.py ./configs/mic/synthiaSeqHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=0 --l-mix-lambda=1.0 --work-dir="./work_dirs/synthia_baseline/synthia_mic_base_short_4gpu02-26-19-05-19" --auto-resume True --wandbid synthia_mic_base_short_4gpu02-26-19-05-19