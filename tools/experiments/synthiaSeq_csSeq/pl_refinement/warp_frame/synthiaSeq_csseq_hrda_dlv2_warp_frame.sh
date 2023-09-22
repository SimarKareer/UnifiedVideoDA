#!/bin/bash
EXP_NAME=synthiaSeq_csSeq_hrda_dlv2_pl_refinement_warp_frame
python ./tools/train.py configs/mic/synthiaSeqHR2csHR_mic_hrda_deeplab.py --no-masking True --launcher=slurm --l-warp-lambda=0.0 --l-mix-lambda=1.0 --warp-cutmix True --bottom-pl-fill True --TPS-warp-pl-confidence True --TPS-warp-pl-confidence-thresh 0.0 --seed 1 --deterministic --work-dir=./work_dirs/synthia/$EXP_NAME --auto-resume True --wandbid $EXP_NAME_$1
