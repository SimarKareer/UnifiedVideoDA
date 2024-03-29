#!/bin/bash
EXP_NAME=viper_csSeq_mic_hrda_pl_refinement_consis
python ./tools/train.py configs/mic/viperHR2csHR_mic_hrda.py --launcher=slurm --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --warp-cutmix True --bottom-pl-fill True --consis-filter True --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
