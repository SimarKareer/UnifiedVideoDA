# dataset settings
# accel setup
im_idx = 0 # curr frame 
imtk_idx = -1 # past frame 
imtktk_idx = -2 # second past frame

FRAME_OFFSET_BDD = [-4, -2, 0]
LOAD_GT = [False, False, True]

bdd_data_root = '/coc/flash9/datasets/bdd100k/videoda-subset'
# Backward
bdd_train_flow_dir= [
    None,
    "/coc/flash9/datasets/bdd100k_flow/t-1_t-2/frame_dist_2/backward/train/images",
    "/coc/flash9/datasets/bdd100k_flow/t_t-1/frame_dist_2/backward/train/images"
]

bdd_val_flow_dir = [
    None,
    "/coc/flash9/datasets/bdd100k_flow/t-1_t-2/frame_dist_2/backward/val/images",
    "/coc/flash9/datasets/bdd100k_flow/t_t-1/frame_dist_2/backward/val_orig_10k/images"
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (720, 720)
ignore_index = [5, 3, 16, 12, 201, 255]


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
    "stub_flow_pipeline": [
        dict(type='LoadFlowFromFileStub'),
    ],
    "shared_pipeline": [
        dict(type='Resize', img_scale=(1280, 720)), # not sure since bdd is 720 x 1280
        dict(type='RandomCrop', crop_size=crop_size),
        dict(type='RandomFlip', prob=0.5),
    ],
    "im_pipeline": [
        # dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
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
        # dict(type='Collect', keys=['img', 'gt_semantic_seg'])#, meta_keys=[]),
    ]
}


data = dict(
    train=dict(
        type='UDADataset',
        source=dict(
            type='BDDSeqDataset',
            data_root=bdd_data_root,
            img_dir='train_orig_10k/images',
            ann_dir='train_orig_10k/labels',
            split='splits/valid_imgs_train.txt',
            pipeline=bdd_train_pipeline,
            frame_offset=FRAME_OFFSET_BDD,
            load_gt = LOAD_GT,
            flow_dir=bdd_train_flow_dir,
            ignore_index=ignore_index,
        ),
        target=dict(
            type='BDDSeqDataset',
            data_root=bdd_data_root,
            img_dir='train_orig_10k/images',
            ann_dir='train_orig_10k/labels',
            split='splits/valid_imgs_train.txt',
            pipeline=bdd_train_pipeline,
            frame_offset=FRAME_OFFSET_BDD,
            load_gt = LOAD_GT,
            flow_dir=bdd_train_flow_dir,
            ignore_index=ignore_index,
        )
    ),
    val=dict(
        type='BDDSeqDataset',
        data_root=bdd_data_root,
        img_dir='val_orig_10k/images',
        ann_dir='val_orig_10k/labels',
        split='splits/valid_imgs_val.txt',
        pipeline=test_pipeline,
        frame_offset=FRAME_OFFSET_BDD,
        load_gt = LOAD_GT,
        flow_dir=bdd_val_flow_dir,
        ignore_index=ignore_index
    ),
    test=dict(
        type='BDDSeqDataset',
        data_root=bdd_data_root,
        img_dir='val_orig_10k/images',
        ann_dir='val_orig_10k/labels',
        split='splits/valid_imgs_val.txt',
        pipeline=test_pipeline,
        frame_offset=FRAME_OFFSET_BDD,
        load_gt = LOAD_GT,
        flow_dir=bdd_val_flow_dir,
        ignore_index=ignore_index
    )
)