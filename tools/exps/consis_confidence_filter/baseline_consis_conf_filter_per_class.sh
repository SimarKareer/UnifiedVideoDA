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
LOAD_FROM="/coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/lwarp/1gbaseline/iter_40000.pth"
srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=1 --l-warp-begin=1500 --l-mix-lambda=0 --warp-cutmix True --bottom-pl-fill True --consis-confidence-filter True --consis-confidence-thresh 0.95 --consis-confidence-per-class-thresh True --optimizer adamw --lr-schedule poly_10_warm --lr 24e-5 --total-iters=15000 --work-dir="./work_dirs/consis_confidence_filter_train/$1$T" --debug-mode True --nowandb True --load-from $LOAD_FROM
# --wandbid $1$T