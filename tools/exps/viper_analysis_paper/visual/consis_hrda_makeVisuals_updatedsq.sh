#!/bin/bash
#SBATCH --job-name=makeVisuals_updatedsq
#SBATCH --output=makeVisuals_updatedsq.out
#SBATCH --error=makeVisuals_updatedsq.err
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
export MASTER_PORT=27198
source ~/.bashrc
conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x

#change begin to 1500
srun -u python -u ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher=slurm --no-masking True --l-warp-lambda=1.0 --l-warp-begin=0 --l-mix-lambda=0.0 --warp-cutmix True --pl-fill True --consis-filter True --load-from /coc/scratch/vvijaykumar6/mmseg/work_dirs/analysisPaper/viper/viper_hrda_consis_backward_1f08-25-14-19-04/latest.pth --seed 1 --total-iters=10 --deterministic --work-dir=./work_dirs/analysisPaper/viper/makeVisuals_updatedsq09-05-22-21-35 --auto-resume True --nowandb True --debug-mode True