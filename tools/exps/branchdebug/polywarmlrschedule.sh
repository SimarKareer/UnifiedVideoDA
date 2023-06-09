#!/bin/bash
#SBATCH --job-name=branchdebug$1
#SBATCH --output=branchdebug$1.out
#SBATCH --error=branchdebug$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a40"
#SBATCH --partition=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude="optimistprime"
#SBATCH --account="hoffman-lab"

export PYTHONUNBUFFERED=TRUE
export MASTER_PORT=$P
source ~/.bashrc
conda activate visvideoda
cd /coc/testnvme/hmaheshwari7/VideoDA/visualization/mmseg

set -x
srun -u python -u ./tools/train.py ./configs/mic/viperHR2csHR_mic_segformer_mm.py --launcher="slurm" --source-only2 True --lr-schedule poly_10 --total-iters=40000 --work-dir="./work_dirs/branchdebug/$1$T" --wandbid $1$T