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

#change begin to 1500
srun -u python -u ./tools/train.py configs/mic/csSeqHR_mic_hrda_source_deeplab.py --launcher=slurm --eval csseq --load-from /coc/scratch/vvijaykumar6/mmseg/work_dirs/analysisPaper/viper/supervised/csSeq_supervised_baseline_source_dlv2_hrda_40k09-05-21-56-36/iter_40000.pth --source-only2 True --seed 1 --deterministic --work-dir=./work_dirs/analysisPaper/viper/supervised/$1$T --auto-resume True --wandbid $1$T