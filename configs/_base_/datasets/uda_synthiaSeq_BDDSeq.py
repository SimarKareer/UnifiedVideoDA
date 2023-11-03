# dataset settings
FRAME_OFFSET = -2
dataset_type = 'SynthiaSeqDataset'
synthia_data_root = '/srv/share4/datasets/SynthiaSeq/SYNTHIA-SEQS-04-DAWN'
bdd_data_root = '/coc/flash9/datasets/bdd100k/videoda-subset'

synthia_train_flow_dir = '/srv/share4/datasets/SynthiaSeq_Flow/frame_dist_1/forward/train/RGB/Stereo_Left/Omni_F'

# Backward
bdd_train_flow_dir= "/coc/flash9/datasets/bdd100k_flow/t_t-1/frame_dist_2/backward/train/images"
bdd_val_flow_dir = "/coc/flash9/datasets/bdd100k_flow/t_t-1/frame_dist_2/backward/val/images"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (720, 720)
ignore_index = [3, 4, 9, 14, 15, 16, 17, 18, 201, 255] 

synthia_train_pipeline = {
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


bdd_train_pipeline = {
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
        dict(type='Resize', img_scale=(1280, 720)), # not sure since bdd is 720 x 1280
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
        dict(type='Resize', keep_ratio=True, img_scale=(1280, 720)),
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
            type='BDDSeqDataset',
            data_root=bdd_data_root,
            img_dir='train/images',
            ann_dir='train/labels',
            split='splits/valid_imgs_train.txt',
            pipeline=bdd_train_pipeline,
            frame_offset=FRAME_OFFSET,
            flow_dir=bdd_train_flow_dir,
            ignore_index=ignore_index,
        )
    ),
    val=dict(
        type='BDDSeqDataset',
        data_root=bdd_data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        split='splits/valid_imgs_val.txt',
        pipeline=test_pipeline,
        frame_offset=FRAME_OFFSET,
        flow_dir=bdd_val_flow_dir,
        ignore_index=ignore_index
    ),
    test=dict(
        type='BDDSeqDataset',
        data_root=bdd_data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        split='splits/valid_imgs_val.txt',
        pipeline=test_pipeline,
        frame_offset=FRAME_OFFSET,
        flow_dir=bdd_val_flow_dir,
        ignore_index=ignore_index
    )
)