#!/bin/bash
EXP_NAME=synthiaSeq_csSeq_hrda_pl_refinement_oracle
python ./tools/train.py configs/mic/synthiaSeqHR2csHR_mic_hrda.py --launcher=slurm --no-masking True --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --warp-cutmix True --bottom-pl-fill True --oracle-mask True --seed 1 --deterministic --work-dir=./work_dirs/synthia/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
