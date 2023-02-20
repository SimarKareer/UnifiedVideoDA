# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# optimizer
# import torch
import os
lr_scale = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

print("SCALING LR BY: ", lr_scale)

optimizer = dict(
    type='AdamW', lr=0.00006 * lr_scale, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()
