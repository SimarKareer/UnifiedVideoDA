from mmseg.datasets.viperSeq import ViperSeqDataset
from mmseg.datasets.viper import ViperDataset
from torch.utils.data import DataLoader
from functools import partial
from mmcv.parallel import collate
from mmseg.core.evaluation.metrics import flow_prop_iou, intersect_and_union
from tools.aggregate_flows.flow.my_utils import palette_to_id, backpropFlow, imageMap, imshow, labelMapToIm, visFlow
import torch
import numpy as np
from tqdm import tqdm
import pdb
from welford import Welford
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg, get_segmentation_error_vis
import matplotlib.pyplot as plt

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    crop_size = (1024, 1024)
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    viper_data_root = '/coc/testnvme/datasets/VideoDA/VIPER'
    viper_train_flow_dir = "/srv/share4/datasets/VIPER_Flowv3/train/flow_occ"

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

    viper = ViperSeqDataset(
        data_root=viper_data_root,
        img_dir='train/img',
        ann_dir='train/cls',
        split='splits/train.txt',
        pipeline=gta_train_pipeline,
        frame_offset=1,
        flow_dir=viper_train_flow_dir,
        no_crash_dataset=True
    )

    # data_loader = DataLoader(viper)
    data_loader = DataLoader(
        viper,
        collate_fn=partial(collate, samples_per_gpu=1),
        num_workers=0,
        shuffle=True
    )

    w = Welford()
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(3 * 2, 3 * 2),
    )

    # # for i, data in enumerate(tqdm(data_loader)):
    # with tqdm(data_loader) as t:
    #     for i, data in enumerate(t):
    #         im, imtk, flow, gt_t, gt_tk = data["img"][0], data["imtk"][0], data["flow"][0], data["gt_semantic_seg"][0], data["imtk_gt_semantic_seg"][0]
    #         # subplotimg(axs[0, 0], im, "im")
    #         # fig.savefig("/coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/visFlow/im0.png")
    #         samples=np.array([im[i].flatten().numpy() for i in range(3)]).T
    #         w.add_all(samples)
    #         # breakpoint()
    #         # if i % 100 == 0:
    #             # print(w.mean, w.var_p)
    #         t.set_description(f"mean: {w.mean}, var: {w.var_p}")
    N = 1000
    means = []
    PHASE="mean"
    non_skipped = 0

    if PHASE == "std":
        with tqdm(data_loader) as t:
            for i, data in enumerate(t):
                if "failed" in data.keys():
                    continue

                im, imtk, flow, gt_t, gt_tk = data["img"][0].numpy(), data["imtk"][0], data["flow"][0], data["gt_semantic_seg"][0], data["imtk_gt_semantic_seg"][0]
                means.append(np.mean(im.reshape(3, -1), axis=1))

                if i >= N:
                    break
                
                t.set_description(f"mean: {np.mean(means, axis=0)} non_skipped: {non_skipped}")
        mu_rgb = np.mean(means, axis=0)  # mu_rgb.shape == (3,)
        print("mu_rgb", mu_rgb)
    else:
        mu_rgb=np.array([238.27737733, 235.72995985, 226.51926128])
        # std_rgb = np.array([37.13001504 38.79420189 47.94346603])

        variances = []
        with tqdm(data_loader) as t:
            for i, data in enumerate(t):
                if "failed" in data.keys():
                    continue

                im, imtk, flow, gt_t, gt_tk = data["img"][0].numpy(), data["imtk"][0], data["flow"][0], data["gt_semantic_seg"][0], data["imtk_gt_semantic_seg"][0]
                var = np.mean((im.reshape(3, -1) - mu_rgb.reshape(3, 1)) ** 2, axis=1)
                variances.append(var)

                if i >= N:
                    break
                
                non_skipped += 1
                t.set_description(f"std: {np.sqrt(np.mean(variances, axis=0))} non_skipped: {non_skipped}")        

        std_rgb = np.sqrt(np.mean(variances, axis=0))  # std_rgb.shape == (3,)
        print(f"std: {std_rgb}")

main()