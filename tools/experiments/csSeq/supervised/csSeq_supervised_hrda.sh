#!/bin/bash
EXP_NAME=csSeq_supervised_hrda
python ./tools/train.py configs/mic/csSeqHR_mic_hrda_supervised.py --launcher=slurm --source-only2 True --seed 1 --deterministic --work-dir=./work_dirs/csSeq/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
