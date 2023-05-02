# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# optimizer
# import torch
import os
lr_scale = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

# print("SCALING LR BY: ", lr_scale)

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)
# optimizer_config = dict()
optimizer_config = None