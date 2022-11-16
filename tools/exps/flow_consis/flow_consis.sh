#!/bin/bash
#SBATCH --job-name=flow_consis$1
#SBATCH --output=flow_consis$1.out
#SBATCH --error=flow_consis$1.err
#SBATCH --gpus-per-task 1  # This gives 1 GPU to each process (or task)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 # The number of processes for slurm to start on each node
#SBATCH --partition=short
#SBATCH --constraint="a40"
export PYTHONUNBUFFERED=TRUE

source ~/.bashrc

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

conda activate mmseg
cd ~/flash/Projects/VideoDA/mmsegmentation

set -x
python tools/test.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/segformer/segformer.b5.1024x1024.viper.160k.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/segformer.b5.1024x1024.viper.160k/iter_48000.pth --eval mIoU pred_pred gt_pred