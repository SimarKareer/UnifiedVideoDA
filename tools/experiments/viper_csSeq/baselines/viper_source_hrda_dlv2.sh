#!/bin/bash
EXP_NAME=viper_source_hrda_dlv2
python ./tools/train.py configs/mic/viperHR2csHR_mic_hrda_deeplab.py --launcher=slurm --source-only2 True --l-warp-lambda=0.0 --l-mix-lambda=1.0 --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
