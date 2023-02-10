#!/bin/bash
#SBATCH --job-name=lwarp_$1
#SBATCH --output=lwarp_$1.out
#SBATCH --error=lwarp_$1.err
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short
#SBATCH --account=hoffman-lab
#SBATCH -x ./excluded.txt


export PYTHONUNBUFFERED=TRUE
export PYTHONBREAKPOINT=0
source ~/.bashrc
conda activate mic3
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
srun -u python -u tools/train.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=1.0 --l-mix-lambda=0.0 --work-dir="./work_dirs/lwarp/$1"

