#!/bin/bash

EXP_NAME=TESTING_synthiaSeq_bddSeq_hrda_video_disrim_pl_refinement_consis$T
python ./tools/train.py configs/mic/synthiaSeqHR2bddHR_mic_hrda_deeplab.py --launcher=slurm --no-masking True --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --adv-scale 1e-1 --warp-cutmix True --bottom-pl-fill True --consis-filter True --seed 1 --deterministic --work-dir=./work_dirs/synthia/$EXP_NAME --auto-resume True --wandbid $EXP_NAME
