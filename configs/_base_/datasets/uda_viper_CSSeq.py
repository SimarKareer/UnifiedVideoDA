# dataset settings, for frame offset select the frames relative to 0 that you want, load_gt is whether the gt exists for that frame, flow_dir is the corresponding flow (None if unavailable)
FRAME_OFFSET = [-2, -1, 0]
LOAD_GT = [False, False, True]


dataset_type = 'ViperSeqDataset'
viper_data_root = '/coc/testnvme/datasets/VideoDA/VIPER'
cs_data_root = '/coc/testnvme/datasets/VideoDA/cityscapes-seq'

viper_train_flow_dir = [
    None,
    "/coc/testnvme/datasets/VideoDA/VIPER_gen_flow/t_neg_1/frame_dist_1/backward/train/img",
    "/coc/testnvme/datasets/VideoDA/VIPER_gen_flow/frame_dist_1/backward/train/img"
]

# Forward (frame dist = 1)
cs_train_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow/forward/train"
cs_val_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow/forward/val"

# for dist 3,6,10
# cs_train_flow_dir = f'/srv/share4/datasets/cityscapes-seq_Flow/flow_test_bed/frame_dist_{FRAME_OFFSET}/forward/train'
# cs_val_flow_dir = f'/srv/share4/datasets/cityscapes-seq_Flow/flow_test_bed/frame_dist_{FRAME_OFFSET}/forward/val'


# Backward
cs_train_flow_dir = [
    None,
    "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/future_flow/tk_1_flows/frame_dist_1/backward/train/",
    "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow_test_bed/frame_dist_1/backward/train"
]
cs_val_flow_dir = [
    None,
    # "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/future_flow/tk_1_flows/frame_dist_1/backward/val",
    None,
    "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow_test_bed/frame_dist_1/backward/val"
]

# Forward
# cs_train_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow/forward/train"
# cs_val_flow_dir = "/coc/testnvme/datasets/VideoDA/cityscapes-seq_Flow/flow/forward/val"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (1024, 1024)
ignore_index = [5, 3, 16, 12, 201, 255]

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
    "stub_flow_pipeline": [
        dict(type='LoadFlowFromFileStub'),
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
        # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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
    "stub_flow_pipeline": [
        dict(type='LoadFlowFromFileStub'),
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
        # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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
    "stub_flow_pipeline": [
        dict(type='LoadFlowFromFileStub'),
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
        # dict(type='Collect', keys=['img', 'gt_semantic_seg'])#, meta_keys=[]),
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
            frame_offset=FRAME_OFFSET,
            load_gt = LOAD_GT,
            flow_dir=viper_train_flow_dir,
        ),
        target=dict(
            type='CityscapesSeqDataset',
            data_root=cs_data_root,
            img_dir='leftImg8bit_sequence/train',
            ann_dir='gtFine/train',
            split='splits/train.txt', # train_2f.txt (frames t, t+1)
            pipeline=cityscapes_train_pipeline,
            frame_offset=FRAME_OFFSET,
            load_gt = LOAD_GT,
            flow_dir=cs_train_flow_dir,
            ignore_index=ignore_index,
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
        load_gt = LOAD_GT,
        flow_dir=cs_val_flow_dir,
        ignore_index=ignore_index
    ),
    test=dict(
        type='CityscapesSeqDataset',
        data_root=cs_data_root,
        img_dir='leftImg8bit_sequence/val',
        ann_dir='gtFine/val',
        split='splits/val.txt',
        pipeline=test_pipeline,
        frame_offset=FRAME_OFFSET,
        load_gt = LOAD_GT,
        flow_dir=cs_val_flow_dir,
        ignore_index=ignore_index
    )
)