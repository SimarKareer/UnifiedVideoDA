from mmseg.datasets.viperSeq import ViperSeqDataset
from mmseg.datasets.cityscapesSeq import CityscapesSeqDataset
from mmseg.datasets.viper import ViperDataset
from torch.utils.data import DataLoader
from functools import partial
from mmcv.parallel import collate
from mmseg.core.evaluation.metrics import flow_prop_iou, intersect_and_union
import torch
import numpy as np
from tqdm import tqdm
import pdb
from welford import Welford
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg, get_segmentation_error_vis
import matplotlib.pyplot as plt
from torchvision import transforms
from get_dataset import get_viper_train, get_viper_val, get_csseq_train, get_csseq_val



def data_loader_test():
    # data_loader = DataLoader(viper)
    dataset = get_viper_train()
    dataset.data_type = "rgb+flowrgbnorm"
    data_loader = DataLoader(
        dataset,
        collate_fn=partial(collate, samples_per_gpu=1),
        num_workers=0,
        shuffle=True
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(3 * 2, 3 * 2),
    )
    invNorm = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), #Using some other dataset mean and std
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
    ])

    for i, data in enumerate(tqdm(data_loader)):
        # img, imtk, flow, gt_t, gt_tk = data["img"][0].numpy(), data["imtk"][0], data["flow"][0], data["gt_semantic_seg"][0], data["imtk_gt_semantic_seg"][0]
        if "failed" not in data:
            breakpoint()
            subplotimg(axs[0, 0], invNorm(data["img"][0]), "Source IM 0")

            # subplotimg(axs[1, 0], data["flowVisNorm"][0], "Source flow Norm 0")
            subplotimg(axs[0, 1], data["flowVis"][0], "Source flow Norm 0")
            fig.savefig(f"./work_dirs/flowRgbVis/Viper{i}.png")

        if i == 20:
            break

def flow_statistics_gt():
    """
    calculate the distribution of flow magnitudes per class based on ground truth
    """
    NUM_ITERS=500
    
    def get_per_class_hist(dataset, num_iters=100):
        data_loader = DataLoader(
            dataset,
            collate_fn=partial(collate, samples_per_gpu=1),
            num_workers=3,
            shuffle=True
        )

        per_class_hist = torch.zeros((19, 100))

        for i, data in enumerate(tqdm(data_loader)):
            img, _, flow, gt_t, _ = data["img"][0].numpy(), data["imtk"][0], data["flow"][0], data["gt_semantic_seg"][0], data["imtk_gt_semantic_seg"][0]

            for j in range(19):
                per_class_hist[j] += torch.histc(flow.norm(dim=0)[(gt_t == j)[0]], bins=100, min=0, max=500)


            if i == num_iters:
                break
        
        return per_class_hist
    
    def stacked_hist_multiclass(per_class_hist1, per_class_hist2):
        """
        visalize a stacked bar chart for each class in per_class_hist1 and per_class_hist2, which clearly shows both the larger and smaller values
        """
        fig, axs = plt.subplots(19, 1, figsize=(10, 100))
        for i, class_name in enumerate(CityscapesSeqDataset.CLASSES):
            axs[i].bar(np.arange(100), per_class_hist1[i], label="Source", alpha=0.5)
            axs[i].bar(np.arange(100), per_class_hist2[i], label="Target", alpha=0.5)
            axs[i].set_title(class_name, fontsize = '25')
            axs[i].legend()
        fig.savefig(f"./work_dirs/flowHistograms/hist.png")
    
    
    viper = get_viper_train()
    csseq = get_csseq_train()
    viper.data_type = "rgb+flow"
    csseq.data_type = "rgb+flow"

    per_class_hist_source, per_class_hist_target = get_per_class_hist(viper, num_iters=NUM_ITERS), get_per_class_hist(csseq, num_iters=NUM_ITERS)
    stacked_hist_multiclass(per_class_hist_source, per_class_hist_target)
    # double_per_class_histograms([2, 1], [1, 2])
    # stacked_histogram([2, 1], [1, 2])



    
def main():
    data_loader_test()

    # flow_statistics_gt()

main()