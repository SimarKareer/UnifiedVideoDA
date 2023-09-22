# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv2red_r50-d8.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/uda_viper_CSSeq_512x512.py',
    # DAFormer Self-Training
    '../_base_/uda/dacs_a999_fdthings_viper.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 2  # seed with median performance
# HRDA Configuration
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        depth=101),
    type='EncoderDecoder',
    decode_head=dict(
        type='DLV2Head'),
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[256, 256],
        crop_size=[512, 512]))
data = dict(
    train=dict(
        # Rare Class Sampling
        # min_crop_ratio=2.0 for HRDA instead of min_crop_ratio=0.5 for
        # DAFormer as HRDA is trained with twice the input resolution, which
        # means that the inputs have 4 times more pixels.
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        # Pseudo-Label Cropping v2 (from HRDA):
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
    # Use one separate thread/worker for data loading.
    workers_per_gpu=3,
    # Batch size
    samples_per_gpu=1,
)
# MIC Parameters
uda = dict(
    # Apply masking to color-augmented target images
    mask_mode='separatetrgaug',
    # Use the same teacher alpha for MIC as for DAFormer
    # self-training (0.999)
    mask_alpha='same',
    # Use the same pseudo label confidence threshold for
    # MIC as for DAFormer self-training (0.968)
    mask_pseudo_threshold='same',
    # Equal weighting of MIC loss
    mask_lambda=1,
    # Use random patch masking with a patch size of 64x64
    # and a mask ratio of 0.7
    l_warp_lambda=1.0,
    l_mix_lambda=0.0,
    consis_filter=False,
    consis_confidence_filter=False,
    consis_confidence_thresh=0,
    consis_confidence_per_class_thresh=False,
    consis_filter_rare_class=False,
    pl_fill=False,
    bottom_pl_fill=False,
    source_only2=False,
    oracle_mask=False,
    warp_cutmix=False,
    stub_training=False,
    l_warp_begin=1500,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True),
    debug_mode=False,
    class_mask_warp=None,
    class_mask_cutmix=None,
    exclusive_warp_cutmix=False,
    modality="rgb",
    modality_dropout_weights=None,
    oracle_mask_add_noise=False,
    oracle_mask_remove_pix=False,
    oracle_mask_noise_percent=0.0,
    TPS_warp_pl_confidence=False,
    TPS_warp_pl_confidence_thresh=0.0,
    max_confidence=False
)
# Optimizer Hyperparameters
optimizer_config = None
# optimizer = dict(
#     lr=6e-05,
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             head=dict(lr_mult=10.0),
#             pos_block=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0))))
# lr_config=None turns off LR schedule
n_gpus = None
launcher = "slurm" #"slurm"
gpu_model = 'A40'
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=1)
evaluation = dict(interval=8000, eval_settings={
    # "metrics": ["mIoU", "pred_pred", "gt_pred", "M5Fixed"],
    "metrics": ["mIoU"],
    "sub_metrics": ["mask_count"],
    "pixelwise accuracy": True,
    "confusion matrix": True,
    "return_logits": False,
    "consis_confidence_thresh": 0.95
})
# Meta Information for Result Analysis
name = 'viperHR2csHR_mic_hrda_s2'
exp = 'basic'
name_dataset = 'viperHR2cityscapesHR_1024x1024'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

# For the other configurations used in the paper, please refer to experiment.py
#CS Invalid Metrics: "M6Sanity", "mIoU_gt_pred", "mIoU"