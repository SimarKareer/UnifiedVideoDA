#!/bin/bash
#SBATCH --job-name=lwarp_$1
#SBATCH --output=lwarp_$1.out
#SBATCH --error=lwarp_$1.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --constraint="a40"
#SBATCH --exclude="clippy,xaea-12"
#SBATCH --partition=short



export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate mic3
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
srun -u python -u tools/train.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=1

