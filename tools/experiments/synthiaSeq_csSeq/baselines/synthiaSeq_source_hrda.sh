#!/bin/bash
EXP_NAME=synthiaSeq_source_hrda
python ./tools/train.py configs/mic/synthiaSeqHR2csHR_mic_hrda.py --launcher=slurm --source-only2 True --l-warp-lambda=0.0 --l-mix-lambda=1.0 --seed 1 --deterministic --work-dir=./work_dirs/synthia/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
