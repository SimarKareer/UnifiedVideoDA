#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=$1.out
#SBATCH --error=$1.err
#SBATCH --gres=gpu:$2
#SBATCH --ntasks=$2
#SBATCH --ntasks-per-node=$2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=long
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
srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher=slurm --no-masking True --l-warp-lambda=1.0 --l-warp-begin=0 --l-mix-lambda=0.0 --warp-cutmix True --pl-fill True --consis-filter True --load-from /coc/scratch/vvijaykumar6/mmseg/work_dirs/analysisPaper/viper/viper_hrda_consis_backward_1f08-25-14-19-04/latest.pth --seed 1 --total-iters=10 --deterministic --work-dir=./work_dirs/analysisPaper/viper/$1$T --auto-resume True --nowandb True --debug-mode True