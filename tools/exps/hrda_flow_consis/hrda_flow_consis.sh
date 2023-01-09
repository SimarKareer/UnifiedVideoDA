#!/bin/bash
#SBATCH --job-name=hrda_flow_consis$1
#SBATCH --output=hrda_flow_consis$1.out
#SBATCH --error=hrda_flow_consis$1.err
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
# python tools/test.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/hrda/hrda.b5.1024x1024.viper.160k.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/HRDA/gtaHR2csHR_hrda_246ef/iter_40000_relevant.pth --eval mIoU
# python tools/test.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/hrda/gtaHR2csHR_hrda.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/HRDA/gtaHR2csHR_hrda_246ef/iter_40000_relevant.pth --eval mIoU --launcher none
python tools/uda_test.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/configs/hrda/gtaHR2csHR_hrda.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/HRDA/gtaHR2csHR_hrda_246ef/iter_40000_relevant.pth --eval mIoU pred_pred gt_pred --sub-metrics mask_count correct_consis --launcher none