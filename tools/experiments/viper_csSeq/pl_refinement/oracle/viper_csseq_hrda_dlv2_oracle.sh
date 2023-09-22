#!/bin/bash
EXP_NAME=viper_csSeq_hrda_dlv2_pl_refinement_oracle
python ./tools/train.py configs/mic/viperHR2csHR_mic_hrda_deeplab.py --no-masking True --launcher=slurm --l-warp-lambda=0.0 --l-mix-lambda=1.0 --warp-cutmix True --bottom-pl-fill True --oracle-mask True --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME --auto-resume True --wandbid $EXP_NAME_$T
