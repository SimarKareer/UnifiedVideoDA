# dataset settings
FRAME_OFFSET = 1
dataset_type = 'ViperSeqDataset'
viper_data_root = '/coc/testnvme/datasets/VideoDA/VIPER'
cs_data_root = '/coc/testnvme/datasets/VideoDA/cityscapes-seq'
cs_train_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow/forward/train"
cs_val_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow/forward/val"
# viper_train_flow_dir = "/srv/share4/datasets/VIPER_Flowv3/train/flow_occ"
viper_train_flow_dir = "/coc/testnvme/datasets/VideoDA/VIPER_gen_flow/frame_dist_1/forward/train/img"

# cs_train_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow_test_bed/frame_dist_10/forward/train"
# cs_val_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow_test_bed/frame_dist_10/forward/val"

# cs_train_flow_dir = f'/srv/share4/datasets/cityscapes-seq_Flow/flow_test_bed/frame_dist_{FRAME_OFFSET}/forward/train'
# cs_val_flow_dir = f'/srv/share4/datasets/cityscapes-seq_Flow/flow_test_bed/frame_dist_{FRAME_OFFSET}/forward/val'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (800, 800)

gta_train_pipeline = {
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
        dict(type='Resize', img_scale=(2560, 1440)),
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
    train=dict(
        type='UDADataset',
        source=dict(
            type='ViperSeqDataset',
            data_root=viper_data_root,
            img_dir='train/img',
            ann_dir='train/cls',
            split='splits/train.txt',
            pipeline=gta_train_pipeline,
            frame_offset=1,
            flow_dir=viper_train_flow_dir,
        ),
        target=dict(
            type='CityscapesSeqDataset',
            data_root=cs_data_root,
            img_dir='leftImg8bit_sequence/train',
            ann_dir='gtFine/train',
            split='splits/train.txt',
            pipeline=cityscapes_train_pipeline,
            frame_offset=FRAME_OFFSET,
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
        frame_offset=FRAME_OFFSET,
        flow_dir=cs_val_flow_dir,
    ),
    test=dict(
        type='CityscapesSeqDataset',
        data_root=cs_data_root,
        img_dir='leftImg8bit_sequence/val',
        ann_dir='gtFine/val',
        split='splits/val.txt',
        pipeline=test_pipeline,
        frame_offset=FRAME_OFFSET,
        flow_dir=cs_val_flow_dir,
    )
)