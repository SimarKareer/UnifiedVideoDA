# dataset settings
dataset_type = 'ViperSeqDataset'
data_root = '/srv/share4/datasets/VIPER/'

#imagenet values
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Flow configs
frame_offset=1

#crop size from the da-vsn paper code
crop_size = (720, 1280)
train_pipeline = { #TOTALLY WRONG
    # "im_load_pipeline": [
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations'),
    # ],
    # "flow_load_pipeline": [
    #     dict(type='LoadImageFromFile'),
    #     dict(type='LoadAnnotations'),
    # ],
    # "shared_pipeline": [
    #     dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    #     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    #     dict(type='RandomFlip', prob=0.5),
    # ],
    # "im_pipeline": [
    #     dict(type='PhotoMetricDistortion'),
    #     dict(type='Normalize', **img_norm_cfg),
    #     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    #     dict(type='DefaultFormatBundle'),
    #     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    # ],
    # "flow_pipeline": [
    #     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    #     # dict(type='DefaultFormatBundle'), I don't know what this is
    #     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    # ]
}

test_pipeline = {
    "im_load_pipeline": [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
    ],
    "load_no_ann_pipeline": [
        dict(type='LoadImageFromFile'),
    ],
    "load_flow_pipeline": [
        dict(type='LoadFlowFromFile'),
    ],
    "shared_pipeline": [
        # dict(type='Resize', keep_ratio=True, img_scale=(1080, 1920)),
        # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.0),
    ],
    "im_pipeline": [
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        # dict(type='DefaultFormatBundle'),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])#, meta_keys=[]),
    ],
    "flow_pipeline": [
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'), #I don't know what this is
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])#, meta_keys=[]),
    ]
}

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=5,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/img',
            ann_dir='train/cls',
            split='splits/train.txt',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/img',
        ann_dir='val/cls',
        split='splits/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/img',
        ann_dir='val/cls',
        split='splits/val.txt',
        pipeline=test_pipeline,
        frame_offset=frame_offset,
        flow_dir="/srv/share4/datasets/VIPER_Flowv3/val/flow_occ",
    )
)
