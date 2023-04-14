#!/bin/bash

set +x
source ~/.bashrc

PARTITION=short
JOB_NAME=$1
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-15}
SRUN_ARGS=""
CURR_DIR=$PWD
DATE_STAMP=$(date +"%m-%d-%Y-%H-%M-%S")
LR=1.2e-4
LR_SCHED="poly_10_warm"
ITERS=1
OPTIM="adamw"
FULL_JOB_NAME=${JOB_NAME}_${DATE_STAMP}_iters_${ITERS}_optim_${OPTIM}_lr_sched_${LR_SCHED}_lr${LR}
echo $FULL_JOB_NAME
LOAD_FROM="/coc/scratch/vvijaykumar6/mmseg/work_dirs/4gpu_baseline/baseline_4gpu_15k_6e-5_04-07-2023-11-52-12_iters_15000_optim_adamw_lr_sched_poly_10_warm_lr6e-5/latest.pth"

PORT=$((40000 + $RANDOM % 1000))
export MASTER_PORT=$PORT
echo "MASTER_PORT: ${PORT}"

cp ${0##*/} $FULL_JOB_NAME.sh

cd /coc/scratch/vvijaykumar6/mmseg
conda activate openmmlab
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --output=${CURR_DIR}/${FULL_JOB_NAME}.out \
    --error=${CURR_DIR}/${FULL_JOB_NAME}.err \
    --account=hoffman-lab \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --constraint="a40" \
    --exclude="clippy,xaea-12,omgwth" \
    python -u ./tools/train.py ./configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=-1 --l-mix-lambda=1 --optimizer $OPTIM --lr-schedule $LR_SCHED --lr=$LR --total-iters=$ITERS --work-dir="./work_dirs/4gpu_baseline/${FULL_JOB_NAME}" --load-from=$LOAD_FROM --nowandb True --eval True