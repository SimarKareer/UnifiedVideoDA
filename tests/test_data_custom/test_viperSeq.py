import pytest
from mmseg.datasets.viperSeq import ViperSeqDataset
from mmseg.datasets.viper import ViperDataset
from torch.utils.data import DataLoader
from functools import partial
from mmcv.parallel import collate
from mmseg.core.evaluation.metrics import flow_prop_iou

def test_viper_seq_shapes():
    # dataset settings
    dataset_type = 'ViperDataset'
    data_root = '/srv/share4/datasets/VIPER/'

    #imagenet values
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    #crop size from the da-vsn paper code
    crop_size = (720, 1280)
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

    

    img_dir = '/srv/share4/datasets/VIPER/'

    viper = ViperSeqDataset(
        data_root=data_root,
        img_dir='val/img',
        ann_dir='val/cls',
        split='splits/val.txt',
        pipeline=test_pipeline,
        frame_offset=1,
        flow_dir="/srv/share4/datasets/VIPER_Flowv3/val/flow_occ"
    )

    data_loader = DataLoader(
        viper,
        collate_fn=partial(collate, samples_per_gpu=1)
    )

    print("made dl")
    for i, data in enumerate(data_loader):
        assert(isinstance(data["img"], list))
        assert(data["img"][0].shape == (1, 3, 1080, 1920))

        assert(isinstance(data["imtk"], list))
        assert(data["imtk"][0].shape == (1, 3, 1080, 1920))

        assert(isinstance(data["gt_semantic_seg"], list))
        assert(data["gt_semantic_seg"][0].shape == (1, 1, 1080, 1920))
        print(data["gt_semantic_seg"][0].shape)

        assert(isinstance(data["imtk_gt_semantic_seg"], list))
        assert(data["imtk_gt_semantic_seg"][0].shape == (1, 1, 1080, 1920))

        assert(isinstance(data["flow"], list))
        assert(data["flow"][0].shape == (1, 2, 1080, 1920))
        break

test_viper_seq_shapes()