#!/bin/bash
#SBATCH --job-name=baseline_base_OR_Filter_testing
#SBATCH --output=baseline_base_OR_Filter_testing.out
#SBATCH --error=baseline_base_OR_Filter_testing.err
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=15
#SBATCH --constraint="a40"
#SBATCH --partition=short
export PYTHONUNBUFFERED=TRUE

source ~/.bashrc

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

conda activate openmmlab
cd /coc/scratch/vvijaykumar6/mmseg

set -x

srun -u python -u tools/train.py ./configs/mic/viperHR2csHR_mic_hrda_eval.py --launcher="slurm" --work-dir="./work_dirs/flow_test_bed/viper_cs-seq_baseline/base_OR_Filter_testing04-05-00-47-35" --auto-resume True --nowandb True --eval True
