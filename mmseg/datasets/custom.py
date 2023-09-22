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

from torch.nn.modules.dropout import _DropoutNd
from timm.models.layers import DropPath
from tools.aggregate_flows.flow.my_utils import backpropFlow, rare_class_or_filter

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
                 load_annotations=True,
                 no_crash_dataset=False
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

        self.no_crash_dataset = no_crash_dataset

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
        if load_annotations:
            self.img_infos = self.load_annotations(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split)

        self.init_cml_metrics()

    def init_cml_metrics(self):
        self.cml_intersect = {k: torch.zeros(len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M5Fixed", "M6", "M6B", "M7", "M8", "M6Sanity", "PL1", "OR_Filter", "inconsis_predt_gt", "inconsis_predtk_gt", "inconsis_predt_predtk", "consis_confidence_filter"]} #TODO: this needs to persist out of this loop for iou prints to be accurate.
        self.cml_union = {k: torch.zeros(len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M5Fixed", "M6", "M6B", "M7", "M8", "M6Sanity", "PL1", "OR_Filter", "inconsis_predt_gt", "inconsis_predtk_gt", "inconsis_predt_predtk", "consis_confidence_filter"]}
        self.mask_counts = {k: torch.zeros(len(self.CLASSES)) for k in ["pred_pred", "gt_pred"]}
        self.total_mask_counts = {k: torch.zeros(len(self.CLASSES)) for k in ["pred_pred", "gt_pred"]}
        self.cml_correct_consis = {k: torch.zeros(len(self.CLASSES)) for k in ["correct_consis", "incorrect_consis", "correct_inconsis", "incorrect_inconsis"]}

        self.pixelwise_correct = {k: torch.zeros(len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M5Fixed", "M6", "M6B", "M7", "M8", "M6Sanity", "PL1", "OR_Filter", "inconsis_predt_gt", "inconsis_predtk_gt", "inconsis_predt_predtk", "consis_confidence_filter"]} #TODO: this needs to persist out of this loop for iou prints to be accurate.
        self.pixelwise_total = {k: torch.zeros(len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M5Fixed", "M6", "M6B", "M7", "M8", "M6Sanity", "PL1", "OR_Filter", "inconsis_predt_gt", "inconsis_predtk_gt", "inconsis_predt_predtk", "consis_confidence_filter"]}

        self.confusion_matrix = {k: torch.zeros(len(self.CLASSES), len(self.CLASSES)) for k in ["mIoU", "mIoU_gt_pred", "pred_pred", "gt_pred", "M5", "M5Fixed", "M6", "M6B", "M7", "M8", "M6Sanity", "PL1", "OR_Filter", "inconsis_predt_gt", "inconsis_predtk_gt", "inconsis_predt_predtk", "consis_confidence_filter"]} #TODO: this needs to persist out of this loop for iou prints to be accurate.
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
                print(f"{'Class':15s}, {'IoU':10s}, {'Intersect':15s}, {'Union':15s}, {'Pixel Accuracy':15s}, {'Correct Pixels':15s}, {'Total Pixels':15s}, {'Mask Ratio':15s}")
            else:
                print(f"{'Class':15s}, {'IoU':10s}, {'Intersect':15s}, {'Union':15s}, {'Pixel Accuracy':15s}, {'Correct Pixels':15s}, {'Total Pixels':15s}")
            # breakpoint()
            ious = self.cml_intersect[metric] / self.cml_union[metric]
            pixel_acc = self.pixelwise_correct[metric] / self.pixelwise_total[metric]
            for i in range(len(self.CLASSES)):
                out_str = ""
                if not np.isnan(ious[i].item()):
                    out_str += f"{self.CLASSES[i]:15s}, {ious[i].item() * 100:05.2f}"
                else:
                    out_str += f"{self.CLASSES[i]:15s}, {'nan':5s}"
                out_str += "     "

                out_str += f", {str(self.cml_intersect[metric][i].item()):15s}, {str(self.cml_union[metric][i].item()):15s}"

                if not np.isnan(pixel_acc[i].item()):
                    out_str += f", {pixel_acc[i].item() * 100:05.2f}"
                else:
                    out_str += f"{', nan  ':5s}"
                out_str += "          "
                out_str += f", {str(self.pixelwise_correct[metric][i].item()):15s}, {str(self.pixelwise_total[metric][i].item()):15s}"

                if "mask_count" in sub_metrics and "mIoU" not in metric and metric != "M5" and metric != "M5Fixed" and metric != "OR_Filter" and metric != "inconsis_predt_gt" and metric != "inconsis_predtk_gt" and metric != "inconsis_predt_predtk" and metric != "consis_confidence_filter":
                    # breakpoint()
                    mask_ratio = 0 if self.total_mask_counts[metric][i] == 0 else self.mask_counts[metric][i] / self.total_mask_counts[metric][i]

                    out_str += f", {100*mask_ratio:.2f}"
                print(out_str)
            
            #print out confusion matrix
            print(f"\nCONFUSION MATRIX FOR EVAL SETTING: {metric}")

            norm = torch.sum(self.confusion_matrix[metric], 1)
            norm = norm.reshape(len(self.CLASSES), 1)
            confusion_matrix_norm = self.confusion_matrix[metric] / norm
            confusion_matrix_norm *= 100
            column_headers = f"{'Classes':15s}"
            for cl in self.CLASSES:
                column_headers += f"{str(cl):15s}"


            formatted_matrix = ""
    
            for i, row in enumerate(confusion_matrix_norm):
                formatted_matrix += f"{str(self.CLASSES[i]):15s}"
                
                for val in row:
                    if not np.isnan(val.item()):
                        formatted_matrix += f"{val.item():05.2f}          "
                    else:
                        formatted_matrix += f"nan            "
                formatted_matrix += "\n"
            print(column_headers)
            print(formatted_matrix)
            
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
    
    def pre_eval_dataloader_consis(self, curr_preds, data, future_preds, metrics=["mIoU"], sub_metrics=[], return_pixelwise_acc=False, return_confusion_matrix=False,out_dir=None, result_logits=None, result_tk_logits=None, consis_confidence_thresh=None):
        assert(curr_preds) is not None

        pre_eval_results = []
        curr_pred = curr_preds[0][:, :, None]

        if future_preds is not None:
            future_pred = future_preds[0][:, :, None]
            future_seg_map = data["imtk_gt_semantic_seg"][0]
            if future_seg_map.shape[0] == 1 and len(future_seg_map.shape) == 4:
                future_seg_map = future_seg_map.squeeze(0).to(future_pred.device)
    
        curr_seg_map = data["gt_semantic_seg"][0]
        
        if curr_seg_map.shape[0] == 1 and len(curr_seg_map.shape) == 4:
            curr_seg_map = curr_seg_map.squeeze(0).to(curr_pred.device)

        flow = None
        if "flow" in data:
            flow = data["flow"][0].squeeze(0).permute((1, 2, 0)).to(curr_pred.device)
        return_mask_count = "mask_count" in sub_metrics
        # breakpoint()
        if "pred_pred" in metrics:
            iau_pred_pred = flow_prop_iou(
                future_pred,
                curr_pred,
                flow,
                num_classes=len(self.CLASSES),
                ignore_index=self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                # indices=indices,
                return_mask_count=return_mask_count,
                preds_t_tk=None,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix
            )
            if return_mask_count:
                iau_pred_pred, mask_count = iau_pred_pred
                self.mask_counts["pred_pred"][mask_count[0]] += mask_count[1]
                self.total_mask_counts["pred_pred"][mask_count[2]] += mask_count[3]
            
            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_pred_pred[-1]
                iau_pred_pred = iau_pred_pred[:-1]

                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["pred_pred"] += pixel_correct
                    self.pixelwise_total["pred_pred"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["pred_pred"] += confusion_matrix
                

            intersection, union, _, _ = iau_pred_pred
            self.cml_intersect["pred_pred"] += intersection
            self.cml_union["pred_pred"] += union
        
        if "gt_pred" in metrics:
            # props pred at t -> ground truth at t-k
            iau_gt_pred = flow_prop_iou(
                future_pred,
                curr_seg_map.permute((1, 2, 0)),
                flow,
                num_classes=len(self.CLASSES),
                ignore_index=self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                # indices=indices,
                return_mask_count=return_mask_count,
                preds_t_tk=None,
                return_mask=True,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix
            )
            if return_mask_count:
                iau_gt_pred, mask_count = iau_gt_pred
                self.mask_counts["gt_pred"][mask_count[0]] += mask_count[1]
                self.total_mask_counts["gt_pred"][mask_count[2]] += mask_count[3]
            
            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_gt_pred[-1]
                iau_gt_pred = iau_gt_pred[:-1]

                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["gt_pred"] += pixel_correct
                    self.pixelwise_total["gt_pred"] += pixel_total

                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["gt_pred"] += confusion_matrix

            intersection, union, _, _, gt_pred_iou_mask = iau_gt_pred
            self.cml_intersect["gt_pred"] += intersection
            self.cml_union["gt_pred"] += union

        if "M5" in metrics:
            consis = (curr_pred == future_pred).squeeze(-1) #where past frame pred matches future frame
            # breakpoint()

            iau = intersect_and_union(
                curr_pred.squeeze(-1), #past frame
                curr_seg_map.squeeze(0), #past GT
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                # indices=indices,
                return_mask=False,
                custom_mask=consis #where past and future agree
            )

            intersection, union, _, _ = iau
            self.cml_intersect["M5"] += intersection
            self.cml_union["M5"] += union
        
        if "M5Fixed" in metrics:
            consis = (curr_pred == backpropFlow(flow, future_pred)).squeeze(-1) #where past frame pred matches future frame
            # breakpoint()

            iau = intersect_and_union(
                curr_pred.squeeze(-1), #past frame
                curr_seg_map.squeeze(0), #past GT
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                # indices=indices,
                return_mask=False,
                custom_mask=consis, #where past and future agree
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau[-1]
                iau = iau[:-1]

                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["M5Fixed"] += pixel_correct
                    self.pixelwise_total["M5Fixed"] += pixel_total

                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["M5Fixed"] += confusion_matrix

            intersection, union, _, _ = iau
            self.cml_intersect["M5Fixed"] += intersection
            self.cml_union["M5Fixed"] += union

        if "mIoU_gt_pred" in metrics:
            iau_miou = intersect_and_union(
                curr_pred.squeeze(-1),
                curr_seg_map.squeeze(0),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                custom_mask = gt_pred_iou_mask,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["mIoU_gt_pred"] += pixel_correct
                    self.pixelwise_total["mIoU_gt_pred"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["mIoU_gt_pred"] += confusion_matrix
                
            intersection, union, _, _ = iau_miou
            self.cml_intersect["mIoU_gt_pred"] += intersection
            self.cml_union["mIoU_gt_pred"] += union
        
        if "OR_Filter" in metrics:
            PL1 = curr_pred
            PL2 = backpropFlow(flow, future_pred)

            PL_OR_Filter = rare_class_or_filter(PL1, PL2)
            iau_miou = intersect_and_union(
                PL_OR_Filter.squeeze(-1),
                curr_seg_map.squeeze(0),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix,
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["OR_Filter"] += pixel_correct
                    self.pixelwise_total["OR_Filter"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["OR_Filter"] += confusion_matrix

            intersection, union, _, _ = iau_miou

            self.cml_intersect["OR_Filter"] += intersection
            self.cml_union["OR_Filter"] += union
        
        if "inconsis_predt_gt" in metrics:
            pred_t = curr_pred
            pred_tk, warp_mask = backpropFlow(flow, future_pred, return_mask=True)

            # warp filter
            pred_t_warp_masked = torch.ones_like(pred_t)*255
            pred_t_warp_masked[warp_mask] = pred_t[warp_mask]
            mask = (pred_t_warp_masked != pred_tk)
            
            #reset 
            pred_t_inconsis = torch.ones_like(pred_t)*255
            pred_t_inconsis[mask] = pred_t_warp_masked[mask]

            iau_miou = intersect_and_union(
                pred_t_inconsis.squeeze(-1),
                curr_seg_map.squeeze(0),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix,
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["inconsis_predt_gt"] += pixel_correct
                    self.pixelwise_total["inconsis_predt_gt"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["inconsis_predt_gt"] += confusion_matrix

            intersection, union, _, _ = iau_miou

            self.cml_intersect["inconsis_predt_gt"] += intersection
            self.cml_union["inconsis_predt_gt"] += union
        
        if "inconsis_predtk_gt" in metrics:
            pred_t = curr_pred
            pred_tk, warp_mask = backpropFlow(flow, future_pred, return_mask=True)

            # warp filter
            pred_t_warp_masked = torch.ones_like(pred_t)*255
            pred_t_warp_masked[warp_mask] = pred_t[warp_mask]
            mask = (pred_t_warp_masked != pred_tk)
            
            #reset 
            pred_tk_inconsis = torch.ones_like(pred_tk)*255
            pred_tk_inconsis[mask] = pred_tk[mask]

            iau_miou = intersect_and_union(
                pred_tk_inconsis.squeeze(-1),
                curr_seg_map.squeeze(0),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix,
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["inconsis_predtk_gt"] += pixel_correct
                    self.pixelwise_total["inconsis_predtk_gt"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["inconsis_predtk_gt"] += confusion_matrix

            intersection, union, _, _ = iau_miou

            self.cml_intersect["inconsis_predtk_gt"] += intersection
            self.cml_union["inconsis_predtk_gt"] += union
        
        if "inconsis_predt_predtk" in metrics:
            pred_t = curr_pred
            pred_tk, warp_mask = backpropFlow(flow, future_pred, return_mask=True)
            
            # warp filter
            pred_t_warp_masked = torch.ones_like(pred_t)*255
            pred_t_warp_masked[warp_mask] = pred_t[warp_mask]
            mask = (pred_t_warp_masked != pred_tk)


            pred_tk_inconsis = torch.ones_like(pred_tk)*255
            pred_t_inconsis = torch.ones_like(pred_t)*255

            pred_tk_inconsis[mask] = pred_tk[mask]
            pred_t_inconsis[mask] = pred_t_warp_masked[mask]

            iau_miou = intersect_and_union(
                pred_t_inconsis.squeeze(-1),
                pred_tk_inconsis.squeeze(-1),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix,
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["inconsis_predt_predtk"] += pixel_correct
                    self.pixelwise_total["inconsis_predt_predtk"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["inconsis_predt_predtk"] += confusion_matrix

            intersection, union, _, _ = iau_miou

            self.cml_intersect["inconsis_predt_predtk"] += intersection
            self.cml_union["inconsis_predt_predtk"] += union
        

        #WIP
        if "consis_confidence_filter" in metrics:
            
            #skip metric if these values are None
            if consis_confidence_thresh is None or result_logits is None:
                raise Exception("consis_confidence_thresh or result_logits is None. These are needed for the consis_confidence_filter metric")
                
            pred_t = curr_pred
            pred_tk, warp_mask = backpropFlow(flow, future_pred, return_mask=True)

            consis_mask = (pred_t == pred_tk)
            # print("consis mask", consis_mask.size(), consis_mask.sum())


            # print("pred Sizes", pred_t.size(), pred_tk.size())

            pred_t_logit = result_logits[0]
            pred_tk_logit = result_tk_logits[0]

            # print("logit Sizes", pred_t_logit.size(), pred_tk_logit.size())


            #get predictions over a max convidence 
            max_logit_val_pred_t , _ = torch.max(pred_t_logit, dim=0)
            # print("max logit", max_logit_val_pred_t.size())
            max_logit_val_pred_t = max_logit_val_pred_t[:,:, None]

            # print("max logit", max_logit_val_pred_t.size())


            confidence_mask = (max_logit_val_pred_t > consis_confidence_thresh)

            # print("Conf mask", confidence_mask.size(), confidence_mask.sum())

            consis_confidence_mask = (confidence_mask & consis_mask).squeeze(-1)
            # print("consis_conf mask", consis_confidence_mask.sum() )

            # print("Mask Size", consis_confidence_mask.size(), consis_confidence_mask.sum())

            #inputs

            # print("Inputs: ", "pred", pred_t.squeeze(-1).size(), "GT", curr_seg_map.squeeze(0).size())

            iau_miou = intersect_and_union(
                pred_t.squeeze(-1),
                curr_seg_map.squeeze(0),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                custom_mask=consis_confidence_mask, #where past and future agree
                reduce_zero_label=self.reduce_zero_label,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix,
            )

            # print("done")

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["consis_confidence_filter"] += pixel_correct
                    self.pixelwise_total["consis_confidence_filter"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["consis_confidence_filter"] += confusion_matrix

            intersection, union, _, _ = iau_miou

            self.cml_intersect["consis_confidence_filter"] += intersection
            self.cml_union["consis_confidence_filter"] += union

        if "mIoU" in metrics:
            iau_miou = intersect_and_union(
                curr_pred.squeeze(-1),
                curr_seg_map.squeeze(0),
                len(self.CLASSES),
                self.ignore_index,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                return_pixelwise_acc=return_pixelwise_acc,
                return_confusion_matrix=return_confusion_matrix,
            )

            #NOW INSERT METRICS FOR PIXEL WISE AND CONFUSION
            if return_pixelwise_acc or return_confusion_matrix:
                other_metrics = iau_miou[-1]
                iau_miou = iau_miou[:-1]
                
                if 'pixelwise accuracy' in other_metrics:
                    pixel_correct, pixel_total = other_metrics['pixelwise accuracy']
                    self.pixelwise_correct["mIoU"] += pixel_correct
                    self.pixelwise_total["mIoU"] += pixel_total
                
                if 'confusion matrix' in other_metrics:
                    confusion_matrix = other_metrics['confusion matrix']
                    self.confusion_matrix["mIoU"] += confusion_matrix

            intersection, union, _, _ = iau_miou
            self.cml_intersect["mIoU"] += intersection
            self.cml_union["mIoU"] += union
            pre_eval_results.append(iau_miou)

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
        """Evaluate the datas

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
