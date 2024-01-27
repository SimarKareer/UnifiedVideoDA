#!/bin/bash
EXP_NAME=synthiaSeq_bddSeq_hrda_dlv2_accel_consis_mixup_video_discrim
python ./tools/train.py configs/mic/synthiaSeqHR2bddHR_mic_hrda_deeplab.py --no-masking True --launcher=slurm --l-warp-lambda=0.0 --l-mix-lambda=1.0 --adv-scale 1e-1 -accel True --seed 1 --deterministic --work-dir=./work_dirs/synthia/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T