# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    "viperHR2csHR_mic_segformer.py"
]
model = dict(
    # For multimodal
    backbone=dict(type='mit_b5_linfus', style='pytorch'),
    multimodal=True,
)
# MIC Parameters
FLOW_TYPE = "rgb+flowxynorm"
uda = dict(
    modality=FLOW_TYPE
)


data = dict(
    train=dict(
        source=dict(
           data_type=FLOW_TYPE
        ),
        target=dict(
           data_type=FLOW_TYPE
        )
    ),
    val=dict(
        data_type=FLOW_TYPE
    ),
    test=dict(
        data_type=FLOW_TYPE
    )
)
evaluation = dict(interval=3000, eval_settings={
    # "metrics": ["mIoU", "multimodalM5", "MM_v1", "MM_v2", "MM_v3"],
    "metrics": ["mIoU", "branch_consis", "branch1_miou", "branch2_miou"],
    "sub_metrics": ["mask_count"],
    "pixelwise accuracy": True,
    "confusion matrix": True,
})