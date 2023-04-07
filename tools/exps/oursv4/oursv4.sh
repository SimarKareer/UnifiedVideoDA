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
FULL_JOB_NAME=${JOB_NAME}_${DATE_STAMP}
echo $FULL_JOB_NAME

PORT=$((40000 + $RANDOM % 1000))
export MASTER_PORT=$PORT
echo "MASTER_PORT: ${PORT}"

cp ${0##*/} $FULL_JOB_NAME.sh

cd /coc/testnvme/skareer6/Projects/VideoDA/experiments/mmsegmentationExps/
conda activate micExp
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
    python -u ./tools/train.py ./configs/mic/viperHR2csHR_mic_hrda.py --launcher="slurm" --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --consis-filter True --no-masking True --warp-cutmix True --bottom-pl-fill True --optimizer adamw --lr-schedule poly_10_warm --lr=6e-5 --total-iters=40000 --work-dir="./work_dirs/oursv3/${FULL_JOB_NAME}" --wandbid=${FULL_JOB_NAME}
