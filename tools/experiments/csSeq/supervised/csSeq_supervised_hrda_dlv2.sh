#!/bin/bash
EXP_NAME=bddSeq_supervised_hrda_dlv2
python ./tools/train.py configs/mic/bddSeqHR_mic_hrda_deeplab_supervised.py --launcher=slurm --source-only2 True --seed 1 --deterministic --work-dir=./work_dirs/csSeq/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
