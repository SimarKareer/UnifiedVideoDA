#!/bin/bash
EXP_NAME=viper_bddSeq_mic_hrda_pl_refinement_consis
python ./tools/train.py configs/mic/viperHR2bddHR_mic_hrda_deeplab.py --launcher=slurm --l-warp-lambda=1.0 --l-warp-begin=1500 --l-mix-lambda=0.0 --warp-cutmix True --bottom-pl-fill True --consis-filter True --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME --auto-resume True --wandbid $EXP_NAME$T
