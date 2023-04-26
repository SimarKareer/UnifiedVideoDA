# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    "viperHR2csHR_mic_hrda.py"
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
    "metrics": ["mIoU"],
    "sub_metrics": ["mask_count"],
    "pixelwise accuracy": True,
    "confusion matrix": True,
})