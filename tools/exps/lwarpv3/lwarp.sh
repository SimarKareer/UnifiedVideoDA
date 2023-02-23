#!/bin/bash
#SBATCH --job-name=lwarpv3_$1
#SBATCH --output=lwarpv3_$1.out
#SBATCH --error=lwarpv3_$1.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --requeue
#SBATCH --open-mode=append

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate mic3
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
srun -u python -u tools/train.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=0.1 --l-mix-lambda=1.0 --work-dir="./work_dirs/lwarpv3/$1$T" --auto-resume True --nowandb True

