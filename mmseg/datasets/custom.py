# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging

import os
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
import torch
from torchvision.utils import save_image


from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations

from mmseg.core.evaluation.metrics import flow_prop_iou, correctness_confusion
from tools.aggregate_flows.flow.my_utils import backpropFlowNoDup

import pdb
@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        label_space: what dataset to use for labels
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                #  load_annotations=True
                #  file_client_args=dict(backend='disk')
                 ):
        if isinstance(pipeline, dict) or isinstance(pipeline, mmcv.utils.config.ConfigDict):
            self.pipeline = {}
            for k, v in pipeline.items():
                self.pipeline[k] = Compose(v)
        else:
            self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette) # NOTE: look out for classes palette name collision
        self.adaptation_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        # self.file_client_args = file_client_args
        # self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        # if load_annotations:
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                            self.ann_dir,
                                            self.seg_map_suffix, self.split)
        
        self.cml_intersect = {k: torch.zeros(len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M6", "M6B", "M7", "M8", "M6Sanity"]} #TODO: this needs to persist out of this loop for iou prints to be accurate.
        self.cml_union = {k: torch.zeros(len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M6", "M6B", "M7", "M8", "M6Sanity"]}
        self.mask_counts = {k: np.zeros(len(self.CLASSES)) for k in ["pred_pred", "gt_pred"]}
        self.total_mask_counts = {k: np.zeros(len(self.CLASSES)) for k in ["pred_pred", "gt_pred"]}
        self.cml_correct_consis = {k: np.zeros(len(self.CLASSES)) for k in ["correct_consis", "incorrect_consis", "correct_inconsis", "incorrect_inconsis"]}


    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        post_pipeline = self.pipeline(results)
        return post_pipeline

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def formatAllMetrics(self, metrics, sub_metrics):
        for metric in metrics:
            print(f"\nEVAL SETTING: {metric}")
            if "mask_count" in sub_metrics and metric != "mIoU":
                print(f"{'Class':15s}, {'IoU':10s}, {'Intersect':15s}, {'Union':15s}, {'Mask Ratio':15s}")
            else:
                print(f"{'Class':15s}, {'IoU':10s}, {'Intersect':15s}, {'Union':15s}")
            # breakpoint()
            ious = self.cml_intersect[metric] / self.cml_union[metric]
            for i in range(len(self.CLASSES)):
                out_str = ""
                if not np.isnan(ious[i].item()):
                    out_str += f"{self.CLASSES[i]:15s}, {ious[i].item() * 100:05.2f}"
                else:
                    out_str += f"{self.CLASSES[i]:15s}, {'nan':5s}"
                out_str += "     "

                out_str += f", {str(self.cml_intersect[metric][i].item()):15s}, {str(self.cml_union[metric][i].item()):15s}"
                if "mask_count" in sub_metrics and "mIoU" not in metric:
                    # breakpoint()
                    mask_ratio = 0 if self.total_mask_counts[metric][i] == 0 else self.mask_counts[metric][i] / self.total_mask_counts[metric][i]

                    out_str += f", {100*mask_ratio:.2f}"
                print(out_str)

        if "correct_consis" in sub_metrics:
            print(f"\nEVAL SETTING: correct_consis")
            print(f"{'Class':15s}, {'correct_consis':15s}, {'correct_inconsis':15s}, {'incorrect_consis':15s}, {'incorrect_inconsis':15s}, {'Percent Consistent':15s}")
            for i in range(len(self.CLASSES)):
                numer = self.cml_correct_consis['correct_consis'][i] + self.cml_correct_consis['incorrect_inconsis'][i]
                denom = self.cml_correct_consis['correct_consis'][i] + self.cml_correct_consis['correct_inconsis'][i] + self.cml_correct_consis['incorrect_consis'][i] + self.cml_correct_consis['incorrect_inconsis'][i]
                # breakpoint()
                print(f"{self.CLASSES[i]:15s}, {str(self.cml_correct_consis['correct_consis'][i].item()):15s}, {str(self.cml_correct_consis['correct_inconsis'][i].item()):15s}, {str(self.cml_correct_consis['incorrect_consis'][i].item()):15s}, {str(self.cml_correct_consis['incorrect_inconsis'][i].item()):15s}, {numer/denom*100:.2f}")
                # a = 10
                # print(f"{self.CLASSES[i]:15s}, {self.cml_correct_consis['correct_consis'][i].item()}")


            # print(f"{self.CLASSES[i]:15s}, {self.mIoU[i]*100:05.2f}, {str(self.intersects[i].item()):10s}, {str(self.unions[i].item()):10s}%")
    
    def formatmIoU(self, miou, intersects=None, unions=None, mask_counts=None, print_na=False):

        # print(f"\n{'-'*100}")
        cml_sum = 0
        count = 0
        idx = 0
        if intersects is not None and unions is not None:
            for val, name, intersect, union in zip(miou, self.CLASSES, intersects, unions):
                val=val.item()
                if not np.isnan(val) or print_na:
                    if mask_counts is None:
                        # print(f"{name:15s}: {val*100:2.2f}    ({intersect}, {union})   ")
                        print(f"{name:15s}, {val*100:05.2f}, {str(intersect.item()):10s}, {str(union.item()):10s}%")
                    else:
                        # masked_nums, total_nums = mask_counts[0][idx], mask_counts[1][idx]
                        mask_ratio = 0 if mask_counts[1][idx] == 0 else mask_counts[0][idx] / mask_counts[1][idx]
                        print(f"{name:15s}, {val*100:05.2f}, {str(intersect.item()):10s}, {str(union.item()):10s}, {100*mask_ratio:.2f}%")
                if not np.isnan(val):
                    cml_sum += val
                    count += 1
                idx += 1
        else:
            for val, name in zip(miou, self.CLASSES):
                val=val.item()
                if not np.isnan(val) or print_na:
                    print(f"{name:15s}, {val*100:2.2f}")
                if not np.isnan(val):
                    cml_sum += val
                    count += 1
        
        # print("HI: ", cml_sum)
        print(f"{'mean':15s}: {cml_sum*100/count:2.2f}")


    def pre_eval_dataloader_consis(self, preds, indices, data, predstk, cached=None, metrics=["mIoU", "pred_pred", "pred_gt", "gt_pred"], sub_metrics=["mask_count", "correct_consis"]):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
            data (dict): dict containing the images and annotations.
            predstk: prediction on frame t-k
            metrics (list[str]): metrics in ["mIoU", "pred_pred", "pred_gt", "gt_pred"]
            sub_metrics (list[str]): eval settings in ["mask_count", "correct_consis"]
            cached dictionary of all necessary cached values to quickly compute metrics {"predt", "predtk", "gt_t", "gt_tk", "pred_t_tk"}

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        assert(preds is not None or cached is not None)
        # In order to compat with batch inference
        if cached is None:
            if not isinstance(indices, list):
                indices = [indices]
            if not isinstance(preds, list):
                preds = [preds]
            if predstk is not None and not isinstance(predstk, list):
                predstk = [predstk]
            assert(len(preds) == 1) #currently only supports batch size 1 cuz of data["gt_semantic_seg"]

        pre_eval_results = []
 
        for i in range(1 if cached else len(preds)):
            if cached is None:
                pred = preds[i][:, :, None]
                index = indices[i]
                return_mask_count = "mask_count" in sub_metrics

                if predstk is not None:
                    predtk = predstk[i][:, :, None]

                seg_map = data["gt_semantic_seg"][0]
                if seg_map.shape[0] == 1 and len(seg_map.shape) == 4:
                    seg_map = seg_map.squeeze(0)
                
                seg_map_tk = data["imtk_gt_semantic_seg"][0]
                if seg_map_tk.shape[0] == 1 and len(seg_map_tk.shape) == 4:
                    seg_map_tk = seg_map_tk.squeeze(0)


                flow = data["flow"][0].squeeze(0).permute((1, 2, 0))

                # breakpoint()
            else:
                # flow = None
                # pred = None
                return_mask_count=False # would need to save the mask counts while caching
                index = cached["index"]

                # breakpoint()

                if cached["pred"].shape[-1] == 1:
                    pred = cached["pred"].numpy()
                    predtk = cached["pred_tk"].numpy()
                    pred_t_tk = cached["pred_t_tk"].numpy()
                else:
                    pred_logits = cached["pred"].numpy()
                    predtk_logits = cached["pred_tk"].numpy()
                    pred_t_tk_logits = cached["pred_t_tk"].numpy()

                    # breakpoint()
                    pred = pred_logits.argmax(-1)[:, :, None]
                    predtk = predtk_logits.argmax(-1)[:, :, None]
                    pred_t_tk = pred_t_tk_logits.argmax(-1)[:, :, None]

                    pred_t_tk_logits_mask = np.any(pred_t_tk_logits == 255, axis=-1)
                    pred_t_tk[pred_t_tk_logits_mask] = 255
                    
                seg_map = cached["gt_t"]
                seg_map_tk = cached["gt_tk"]

                base_mask = (predtk != 255) & (predtk != 201) & (pred_t_tk != 255) & (pred_t_tk != 201)
                base_mask = base_mask.squeeze(-1)


            if "correct_cons" in sub_metrics:
                cons_correct_dict = correctness_confusion(
                    seg_map_tk.permute((1, 2, 0)),
                    None if cached else pred,
                    predtk,
                    None if cached else flow,
                    self.label_map,
                    preds_t_tk=pred_t_tk
                )
                for k, v in cons_correct_dict.items():
                    dict_to_arr = np.zeros(len(self.CLASSES))
                    for k2, v2 in v.items():
                        if k2 > len(self.CLASSES):
                            # breakpoint()
                            # raise AssertionError("class numbers are mismatching")
                            continue
                        dict_to_arr[k2] = v2
                    self.cml_correct_consis[k] += dict_to_arr
                # breakpoint()

            if "pred_pred" in metrics:
                iau_pred_pred = flow_prop_iou(
                    None if cached else pred,
                    predtk,
                    None if cached else flow,
                    num_classes=len(self.CLASSES),
                    ignore_index=self.ignore_index,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    indices=indices,
                    return_mask_count=return_mask_count,
                    preds_t_tk=pred_t_tk
                )
                if return_mask_count:
                    iau_pred_pred, mask_count = iau_pred_pred
                    self.mask_counts["pred_pred"][mask_count[0]] += mask_count[1]
                    self.total_mask_counts["pred_pred"][mask_count[2]] += mask_count[3]


                intersection, union, _, _ = iau_pred_pred
                self.cml_intersect["pred_pred"] += intersection
                self.cml_union["pred_pred"] += union
            
            if "gt_pred" in metrics:
                # props pred at t -> ground truth at t-k
                iau_gt_pred = flow_prop_iou(
                    None if cached else pred,
                    seg_map_tk.permute((1, 2, 0)),
                    None if cached else flow,
                    num_classes=len(self.CLASSES),
                    ignore_index=self.ignore_index,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    indices=indices,
                    return_mask_count=return_mask_count,
                    preds_t_tk=pred_t_tk,
                    return_mask=True
                )
                if return_mask_count:
                    iau_gt_pred, mask_count = iau_gt_pred
                    self.mask_counts["gt_pred"][mask_count[0]] += mask_count[1]
                    self.total_mask_counts["gt_pred"][mask_count[2]] += mask_count[3]

                intersection, union, _, _, gt_pred_iou_mask = iau_gt_pred
                self.cml_intersect["gt_pred"] += intersection
                self.cml_union["gt_pred"] += union

                # self.gt_pred_iou_mask = gt_pred_iou_mask
            
            if "M5" in metrics:
                consis = (predtk == pred_t_tk).squeeze(-1) #where past frame pred matches future frame
                # breakpoint()

                iau = intersect_and_union(
                    predtk.squeeze(-1), #past frame
                    seg_map_tk.squeeze(0), #past GT
                    len(self.CLASSES),
                    self.ignore_index,
                    label_map=self.label_map,
                    indices=indices,
                    return_mask=False,
                    custom_mask=consis #where past and future agree
                )

                intersection, union, _, _ = iau
                self.cml_intersect["M5"] += intersection
                self.cml_union["M5"] += union
            
            if "M6" in metrics:
                consis = (predtk != pred_t_tk).squeeze(-1) #where past frame doesn't match future frame
                # breakpoint()
                consis = consis & base_mask

                iau = intersect_and_union(
                    pred_t_tk.squeeze(-1), #future frame
                    seg_map_tk.squeeze(0), #past GT
                    len(self.CLASSES),
                    self.ignore_index,
                    label_map=self.label_map,
                    indices=indices,
                    return_mask=False,
                    custom_mask=consis # future and past don't agree
                )

                intersection, union, _, _ = iau
                self.cml_intersect["M6"] += intersection
                self.cml_union["M6"] += union
            
            # M6 Sanity check.  Same as M6 but with ground truth of t
            if "M6Sanity" in metrics:
                # breakpoint()
                pred_t_tk = backpropFlowNoDup(flow.numpy(), pred)

                inconsis = (predtk != pred_t_tk).squeeze(-1)

                base_mask = (predtk != 255) & (predtk != 201) & (pred_t_tk != 255) & (pred_t_tk != 201)
                base_mask = base_mask.squeeze(-1)

                inconsis = inconsis & base_mask

                gt_t_tk = backpropFlowNoDup(flow.numpy(), seg_map.numpy().transpose((1, 2, 0)))


                iau = intersect_and_union(
                    gt_t_tk.squeeze(-1), #past frame
                    seg_map_tk.squeeze(0), #past GT
                    len(self.CLASSES),
                    self.ignore_index,
                    label_map=self.label_map,
                    indices=indices,
                    return_mask=False,
                    custom_mask=inconsis # future and past don't agree
                )

                intersection, union, _, _ = iau
                self.cml_intersect["M6Sanity"] += intersection
                self.cml_union["M6Sanity"] += union




            
            if "M6B" in metrics:
                consis = (predtk != pred_t_tk).squeeze(-1) #where past frame doesn't match future frame
                # breakpoint()
                consis = consis & base_mask

                iau = intersect_and_union(
                    predtk.squeeze(-1), #past frame
                    seg_map_tk.squeeze(0), #past GT
                    len(self.CLASSES),
                    self.ignore_index,
                    label_map=self.label_map,
                    indices=indices,
                    return_mask=False,
                    custom_mask=consis # future and past don't agree
                )

                intersection, union, _, _ = iau
                self.cml_intersect["M6B"] += intersection
                self.cml_union["M6B"] += union
            
            if "M7" in metrics:
                low_conf = predtk_logits.max(axis=-1) < 0.5

                iau = intersect_and_union(
                    pred_t_tk.squeeze(-1), #future frame
                    seg_map_tk.squeeze(0), #past GT
                    len(self.CLASSES),
                    self.ignore_index,
                    label_map=self.label_map,
                    indices=indices,
                    return_mask=False,
                    custom_mask=low_conf # future and past don't agree
                )

                intersection, union, _, _ = iau
                self.cml_intersect["M7"] += intersection
                self.cml_union["M7"] += union

                # breakpoint()
            
            if "M8" in metrics:
                pred_t_tk_logits_mask = np.logical_not(np.any(pred_t_tk_logits == 255, axis=-1)) # uses a slightly weak property, that since this is a post SM logit, the only way a value could be 255 is if i put it there.
                ensemble = ((predtk_logits + pred_t_tk_logits) / 2).argmax(axis=-1)[:, :, None]
                # breakpoint()
                # Bug: This isn't accounting for the masked out pixels of pred_t_tk_logits

                iau = intersect_and_union(
                    ensemble.squeeze(-1),
                    seg_map_tk.squeeze(0), #past GT
                    len(self.CLASSES),
                    self.ignore_index,
                    label_map=self.label_map,
                    indices=indices,
                    return_mask=False,
                    custom_mask=pred_t_tk_logits_mask
                )

                intersection, union, _, _ = iau
                self.cml_intersect["M8"] += intersection
                self.cml_union["M8"] += union
            
            if "mIoU_gt_pred" in metrics:
                iau_miou = intersect_and_union(
                    pred.squeeze(-1),
                    seg_map.squeeze(0),
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    indices=indices,
                    custom_mask = gt_pred_iou_mask
                )
                intersection, union, _, _ = iau_miou
                self.cml_intersect["mIoU_gt_pred"] += intersection
                self.cml_union["mIoU_gt_pred"] += union
            
            if "mIoU" in metrics:
                # print("got mIoU")
                
                iau_miou = intersect_and_union(
                    pred.squeeze(-1),
                    seg_map.squeeze(0),
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    indices=indices
                )
                intersection, union, _, _ = iau_miou
                self.cml_intersect["mIoU"] += intersection
                self.cml_union["mIoU"] += union
                pre_eval_results.append(iau_miou)

            if index % 50 == 0:
                self.formatAllMetrics(metrics=metrics, sub_metrics=sub_metrics)
                # for key in [k for k in metrics if k != "mask_count"]:
                #     intersection = self.cml_intersect[key]
                #     union = self.cml_union[key]
                #     print(f"{'-'*100}\n{key}: ")
                #     if return_mask_count and key in ["pred_pred", "gt_pred"]:
                #         self.formatmIoU(intersection / union, self.cml_intersect[key], self.cml_union[key], mask_counts=(self.mask_counts[key], self.total_mask_counts[key]))
                #     else:
                #         self.formatmIoU(intersection / union, self.cml_intersect[key], self.cml_union[key])

        return pre_eval_results

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    indices=indices))

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 label_map=dict(),
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            print("results: ", results)
            print("seg_maps: ", gt_seg_maps)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
