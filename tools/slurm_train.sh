#!/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-15}
SRUN_ARGS=""

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
    --exclude="clippy,xaea-12" \
    python -u tools/train.py ${CONFIG} --launcher="slurm" --l-warp-lambda=1 --work-dir="./work_dirs/lwarp/$2" --nowandb True
