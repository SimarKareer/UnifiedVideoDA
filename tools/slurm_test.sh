#!/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-15}
# PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --account=${ACCOUNT} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --constraint="a40" \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --eval mIoU pred_pred gt_pred M5 mIoU_gt_pred


#GPUS=2 GPUS_PER_NODE=2 CPUS_PER_TASK=15 ACCOUNT=overcap sh tools/slurm_train.sh overcap backward_flow /coc/scratch/vvijaykumar6/mmseg/configs/viperHR2csHR_mic_hrda_eval.py /coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/lwarp/1gbaseline/iter_40000.pth
