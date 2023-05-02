#!/bin/bash
#SBATCH --job-name=baseline_base_inconsis_metrics2
#SBATCH --output=baseline_base_inconsis_metrics2.out
#SBATCH --error=baseline_base_inconsis_metrics2.err
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

srun -u python -u tools/train.py ./configs/mic/viperHR2csHR_mic_hrda_eval.py --launcher="slurm" --work-dir="./work_dirs/flow_test_bed/viper_cs-seq_baseline/base_inconsis_metrics204-06-00-26-18" --auto-resume True --nowandb True --eval True
