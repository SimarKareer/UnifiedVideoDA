#!/bin/bash
EXP_NAME=viper_csSeq_hrda_dlv2_no_MRFusion_no_rcs_no_imnet
python ./tools/train.py configs/mic/viperHR2csHR_mic_deeplab_no_MRFusion_no_rcs.py --launcher=slurm --no-masking True --l-warp-lambda=0.0 --l-mix-lambda=1.0 --imnet-feature-dist-lambda=0.0 --seed 1 --deterministic --work-dir=./work_dirs/viper/$EXP_NAME --auto-resume True --wandbid $EXP_$T