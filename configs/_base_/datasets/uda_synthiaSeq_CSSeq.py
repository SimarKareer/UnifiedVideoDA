# dataset settings
dataset_type = 'SynthiaSeqDataset'
synthia_data_root = '/srv/share4/datasets/SynthiaSeq/SYNTHIA-SEQS-04-DAWN'
cs_data_root = '/coc/testnvme/datasets/VideoDA/cityscapes-seq'
cs_train_flow_dir = '/srv/share4/datasets/cityscapes-seq_Flow/flow/forward/train'

synthia_train_flow_dir = '/srv/share4/datasets/SynthiaSeq_Flow/frame_dist_1/forward/train/RGB/Stereo_Left/Omni_F'
cs_val_flow_dir = '/srv/share4/datasets/cityscapes-seq_Flow/flow/forward/val'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (1024, 1024)

synthia_train_pipeline = {
    "im_load_pipeline": [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
    ],
    "load_no_ann_pipeline": [
        dict(type='LoadImageFromFile'),
    ],
    "load_flow_pipeline": [
        dict(type='LoadFlowFromFileStub'),
    ],
    "shared_pipeline": [
        dict(type='Resize', img_scale=(2560, 1520)), 
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
    ],
    "im_pipeline": [
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        # dict(type='DefaultFormatBundle'), #I'm not sure why I had to comment it for im, but not for flow.
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ],
    "flow_pipeline": [
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'), #I don't know what this is
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
}

# cityscapes_train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(2048, 1024)),
#     dict(type='RandomCrop', crop_size=crop_size),
#     dict(type='RandomFlip', prob=0.5),
#     # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

cityscapes_train_pipeline = {
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
        dict(type='Resize', img_scale=(2048, 1024)),
        dict(type='RandomCrop', crop_size=crop_size),
        dict(type='RandomFlip', prob=0.5),
    ],
    "im_pipeline": [
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        # dict(type='DefaultFormatBundle'), 
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ],
    "flow_pipeline": [
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
}

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 1024),
#         # MultiScaleFlipAug is disabled by not providing img_ratios and
#         # setting flip=False
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

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
        dict(type='Resize', keep_ratio=True, img_scale=(2048, 1024)),
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
        # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'), #I don't know what this is
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])#, meta_keys=[]),
    ]
}


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SynthiaSeqDataset',
            data_root=synthia_data_root,
            img_dir='RGB/Stereo_Left/Omni_F',
            ann_dir='GT/LABELS/Stereo_Left/Omni_F',
            split='splits/flow/forward/train.txt',
            pipeline=synthia_train_pipeline,
            frame_offset=1,
            flow_dir=synthia_train_flow_dir, 
        ),
        target=dict(
            type='CityscapesSeqDataset',
            data_root=cs_data_root,
            img_dir='leftImg8bit_sequence/train',
            ann_dir='gtFine/train',
            split='splits/train.txt',
            pipeline=cityscapes_train_pipeline,
            frame_offset=1,
            flow_dir=cs_train_flow_dir,
        )
    ),
    val=dict(
        type='CityscapesSeqDataset',
        data_root=cs_data_root,
        img_dir='leftImg8bit_sequence/val',
        ann_dir='gtFine/val',
        split='splits/val.txt',
        pipeline=test_pipeline,
        frame_offset=1,
        flow_dir=cs_val_flow_dir
    ),
    test=dict(
        type='CityscapesSeqDataset',
        data_root=cs_data_root,
        img_dir='leftImg8bit_sequence/val',
        ann_dir='gtFine/val',
        split='splits/val.txt',
        pipeline=test_pipeline,
        frame_offset=1,
        flow_dir=cs_val_flow_dir
    )
)
