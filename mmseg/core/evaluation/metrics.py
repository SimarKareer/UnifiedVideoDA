# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from torchvision.utils import save_image
import cv2
import linecache
import os
from tools.aggregate_flows.flow.my_utils import palette_to_id
from tools.aggregate_flows.flow.my_utils import imshow, visFlow, loadVisFlow, loadFlow, mergeFlow, backpropFlow, imageMap, labelMapToIm, backpropFlowNoDup, palette_to_id
import matplotlib.pyplot as plt


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score

def error_viz(pred_label, label, indices, split="/srv/share4/datasets/VIPER/splits/val.txt"):
    if len(pred_label.shape) != 2:
        raise NotImplementedError("Error viz only works with batch size 1 currently")

    source_im_path = linecache.getline(split, indices[0] + 1).strip() #"/srv/share4/datasets/VIPER/val/img/001/001_00001.jpg"
    source_im = mmcv.imread(f"/srv/share4/datasets/VIPER/val/img/{source_im_path}.jpg", flag='unchanged', backend='pillow')
    for classId in range(32):
        print("making dir")
        if not os.path.exists(f"work_dirs/ims/error_vis/cls{classId}"):
            os.mkdir(f"work_dirs/ims/error_vis/cls{classId}")
    for classId in range(32): #let's look at sidewalk errors
        # print(type(label))
        # print(type(pred_label))
        # print("Label shape: ", label.shape)

        # out_image = label.numpy().copy()[None, :, :].repeat(3, axis=0)
        out_image = np.transpose(source_im.copy(), (2, 0, 1))

        # print("Out image shape: ", out_image.shape)
        # print("l=i", (label==i).shape)
        # print("predl=i", (pred_label==i).shape)
        labeln = label.numpy()
        pred_labeln = pred_label.numpy()
        tp = np.logical_and(labeln == classId, pred_labeln==classId).astype(bool)
        fp = np.logical_and(labeln != classId, pred_labeln==classId).astype(bool)
        fn = np.logical_and(labeln == classId, pred_labeln!=classId).astype(bool)
        # print("TP shape: ", tp.shape)
        # print(tp)
        # print("Out image shape: ", out_image.shape)
        out_image[:, tp] = np.array([0, 255, 0]).reshape(3, 1)
        out_image[:, fp] = np.array([255, 0, 0]).reshape(3, 1)
        out_image[:, fn] = np.array([0, 0, 255]).reshape(3, 1)
        # print("indices: ", indices)
        # print(f"workdirs/ims/pred_errors_t={indices[0]}class={classId}.png")
        # print(f"{out_image.shape=}")
        cv2.imwrite(f"work_dirs/ims/error_vis/cls{classId}/t={indices[0]}.png", np.transpose(out_image, (1, 2, 0)))

def check_pred_shapes(listOfPreds):
    """
    listOfPreds (list) [np.array(), ...]
    """
    for i in listOfPreds:
        assert len(i.shape) == 3 and i.shape[2] < 10, f"pred appears to be the wrong shape.  Got {i.shape}"

def flow_prop_iou(gt_t, gt_tk, flow_tk_t, num_classes=31, return_mask_count=False, preds_t_tk=None, **kwargs):
    """
    gt_t: H, W, *.  image at time t
    gt_tk: H, W, *  image at t-k
    flow_tk_t: H, W, 2 Flow from t-k to t

    If you pass in a cached value for preds_t_tk you can ignore gt_t and flow_tk_t
    """

    if preds_t_tk is None:
        assert len(gt_t.shape) == 3 and gt_t.shape[2] < 10, f"gt_t appears to be the wrong shape.  Got {gt_t.shape}"
        assert len(gt_tk.shape) == 3 and gt_tk.shape[2] < 10, f"gt_tk appears to be the wrong shape.  Got {gt_tk.shape}"
        assert len(flow_tk_t.shape) == 3 and flow_tk_t.shape[2] < 10, f"flow_tk_t appears to be the wrong shape.  Got {flow_tk_t.shape}"

        assert isinstance(gt_t, torch.Tensor), f"gt_t is not a torch tensor.  Got {type(gt_t)}"
        assert isinstance(gt_tk, torch.Tensor), f"gt_tk is not a torch tensor.  Got {type(gt_tk)}"
        assert isinstance(flow_tk_t, torch.Tensor), f"flow_tk_t is not a torch tensor.  Got {type(flow_tk_t)}"

        if return_mask_count:
            mlabel2_1, mask_count = backpropFlow(flow_tk_t, gt_t, return_mask_count=return_mask_count)
        else:
            mlabel2_1 = backpropFlow(flow_tk_t, gt_t)
    else:
        assert len(gt_tk.shape) == 3 and gt_tk.shape[2] < 10, f"gt_tk appears to be the wrong shape.  Got {gt_tk.shape}"
        assert gt_t is None, "Got a value for gt_t but preds_t_tk is not None.  This is not supported."
        assert flow_tk_t is None, "Got a value for flow_tk_t but preds_t_tk is not None.  This is not supported."

        gt_tk = gt_tk.numpy() if isinstance(gt_tk, torch.Tensor) else gt_tk
        mlabel2_1 = preds_t_tk
    # viz = labelMapToIm(torch.tensor(mlabel2_1).long(), palette_to_id).numpy().astype(np.int16)

    # imshow(viz, scale=0.5)
    iau = intersect_and_union(gt_tk.squeeze(2), mlabel2_1.squeeze(2), num_classes=num_classes, **kwargs)
    # print(t.shape, tk.shape, flow_tk_t.shape)

    if return_mask_count:
        mask1 = mask_count[0] < num_classes
        mask_count[0] = mask_count[0][mask1]
        mask_count[1] = mask_count[1][mask1]

        mask2 = mask_count[2] < num_classes
        mask_count[2] = mask_count[2][mask2]
        mask_count[3] = mask_count[3][mask2]


    if return_mask_count:
        return iau, mask_count
    else:
        return iau

def correctness_confusion(gt_tk, pred_t, pred_tk, flow_tk_t, label_map, preds_t_tk=None, **kwargs):
    """
    gt_tk: H, W, 1  ground truth at t-k
    pred_t: H, W, 1  image at t
    pred_tk: H, W, 1  image at t-k
    flow_tk_t: H, W, 2 Flow from t-k to t
    label_map: (dict) label map at t

    if you give a value for preds_t_tk you can ignore pred_t and flow_tk_t
    """
    assert len(gt_tk.shape) == 3 and gt_tk.shape[2] < 10, f"gt_tk appears to be the wrong shape.  Got {gt_tk.shape}"
    assert len(pred_tk.shape) == 3 and pred_tk.shape[2] < 10, f"pred_tk appears to be the wrong shape.  Got {pred_tk.shape}"
    if preds_t_tk is None:
        assert len(pred_t.shape) == 3 and pred_t.shape[2] < 10, f"pred_t appears to be the wrong shape.  Got {pred_t.shape}"
        assert len(flow_tk_t.shape) == 3 and flow_tk_t.shape[2] < 10, f"flow_tk_t appears to be the wrong shape.  Got {flow_tk_t.shape}"
    else:
        assert pred_t is None, "Got a value for pred_t but preds_t_tk is not None.  This is not supported."
        assert flow_tk_t is None, "Got a value for flow_tk_t but preds_t_tk is not None.  This is not supported."

    gt_tk = gt_tk.numpy() if isinstance(gt_tk, torch.Tensor) else gt_tk
    pred_tk = pred_tk.numpy() if isinstance(pred_tk, torch.Tensor) else pred_tk
    if preds_t_tk is None:
        pred_t = pred_t.numpy() if isinstance(pred_t, torch.Tensor) else pred_t
        flow_tk_t = flow_tk_t.numpy() if isinstance(flow_tk_t, torch.Tensor) else flow_tk_t

    if label_map is not None:
        label_copy = gt_tk.clone()
        for old_id, new_id in label_map.items():
            gt_tk[label_copy == old_id] = new_id

    # first find the pixels that are correct in the current frame
    correct = gt_tk == pred_tk
    incorrect = gt_tk != pred_tk
    # of these pixels, find the ones which are consistent vs inconsistent between frames via flow
    if preds_t_tk is None:
        mlabel2_1 = backpropFlow(flow_tk_t, pred_t)
    else:
        mlabel2_1 = preds_t_tk

    consistent_correct = np.logical_and(correct, (mlabel2_1 == pred_tk))
    consistent_incorrect = np.logical_and(incorrect, (mlabel2_1 == pred_tk))
    inconsistent_correct = np.logical_and(correct, (mlabel2_1 != pred_tk))
    inconsistent_incorrect = np.logical_and(incorrect, (mlabel2_1 != pred_tk))

    # return counts by class

    # Same with incorrect pixels
    consistent_correct_count = np.unique(pred_tk[consistent_correct], return_counts=True)
    consistent_correct_count = {k: v for k, v in zip(consistent_correct_count[0], consistent_correct_count[1])}
    consistent_incorrect_count = np.unique(pred_tk[consistent_incorrect], return_counts=True)
    consistent_incorrect_count = {k: v for k, v in zip(consistent_incorrect_count[0], consistent_incorrect_count[1])}
    inconsistent_correct_count = np.unique(pred_tk[inconsistent_correct], return_counts=True)
    inconsistent_correct_count = {k: v for k, v in zip(inconsistent_correct_count[0], inconsistent_correct_count[1])}
    inconsistent_incorrect_count = np.unique(pred_tk[inconsistent_incorrect], return_counts=True)
    inconsistent_incorrect_count = {k: v for k, v in zip(inconsistent_incorrect_count[0], inconsistent_incorrect_count[1])}

    return {"correct_consis": consistent_correct_count, "correct_inconsis": inconsistent_correct_count, "incorrect_consis": consistent_incorrect_count, "incorrect_inconsis": inconsistent_incorrect_count}

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        error_viz_split=None,
                        label_map=dict(),
                        reduce_zero_label=False,
                        indices=None,
                        return_mask=False,
                        return_confusion_matrix=False,
                        return_pixelwise_acc=False,
                        custom_mask=None):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename (H, W).
        label (ndarray | str): Ground truth segmentation map
            or label filename (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.
        return_mask (bool): Whether return mask. Default: False.
        custom_mask (ndarray): Will calc IoU over only pixels where mask == True. Default: None.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """
    # print("shapes: ", pred_label.shape)
    assert len(pred_label.shape) == 2, f"pred_label has wrong dimension.  Got{pred_label.shape}"
    assert len(label.shape) == 2, f"label has wrong dimension.  Got{label.shape}"
    
    
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    elif isinstance(pred_label, np.ndarray):
        pred_label = torch.from_numpy((pred_label))
    
    if isinstance(ignore_index, int):
        ignore_index = [ignore_index]

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    elif isinstance(label, np.ndarray):
        label = torch.from_numpy(label)

    if error_viz_split is not None:
        error_viz(pred_label, label, indices, split=error_viz_split)

    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    # pred = torch.tensor(pred_label).view((1080, 1920, 1))
    # gt = torch.tensor(label).view((1080, 1920, 1)).long()
    # colored_pred = labelMapToIm(pred, palette_to_id)
    # colored_gt = labelMapToIm(gt, palette_to_id)
    # cv2.imwrite("work_dirs/ims/metricsPred.png", colored_pred.numpy().astype(np.int16))
    # cv2.imwrite("work_dirs/ims/metricsLabel.png", colored_gt.numpy().astype(np.int16))
    # cv2.imwrite("work_dirs/ims/metricsPred2.png", pred.numpy().astype(np.int16))
    # cv2.imwrite("work_dirs/ims/metricsLabel2.png", gt.numpy().astype(np.int16))

    def ignore_indices(mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        for idx in ignore_index:
            mask = torch.logical_and(mask, pred_label != idx)
            mask = torch.logical_and(mask, label != idx)
        return mask

    if custom_mask is not None:
        mask = custom_mask
        mask = ignore_indices(mask)
    else:
        mask = torch.ones_like(label, dtype=torch.bool)
        mask = ignore_indices(mask)

        # mask = torch.logical_and(mask, label != 201)
        # mask = torch.logical_and(mask, pred_label != ignore_index)
        # mask = torch.logical_and(mask, pred_label != 201)
    # for ignore in ignore_index:

    # print("shape: ", mask.shape, "masked: ", mask.sum())
    # print("before: ", pred_label.shape, label.shape)
    # breakpoint()
    pred_label = pred_label[mask]
    label = label[mask]
    # print("after: ", pred_label.shape, label.shape)


    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect

    # print(area_intersect / area_union)
    to_return = [area_intersect.cpu(), area_union.cpu(), area_pred_label.cpu(), area_label.cpu()]
    if return_mask:
        to_return.append(mask)
    

    # other metrics
    other_metrics = {}

    if return_confusion_matrix or return_pixelwise_acc:
        matrix = per_class_pixel_accuracy(pred_label.cpu(), label.cpu(),ignore_index=ignore_index,return_raw=True)

        if return_confusion_matrix:
            other_metrics["confusion matrix"] = matrix
        if return_pixelwise_acc:
            pixel_acc_hit = matrix.diagonal()
            pixel_acc_total = matrix.sum(axis=1)
            other_metrics["pixelwise accuracy"] = (pixel_acc_hit, pixel_acc_total)


    
    if return_confusion_matrix or return_pixelwise_acc:
        to_return.append(other_metrics)


    return (*to_return,)

    # if return_mask:
    #     return area_intersect, area_union, area_pred_label, area_label, mask
    # else:
    #     return area_intersect, area_union, area_pred_label, area_label

def ignore_indices(image, ignore_index):
    """
    Given image: (H, W) and indices: (N), returns a mask of shape (H, W) which is True wherever image is not in indices
    """
    mask = torch.ones_like(image, dtype=torch.bool)

    for idx in ignore_index:
        mask = torch.logical_and(mask, image != idx)
    return mask

# used for training metrics 
def per_class_pixel_accuracy(pred, label, return_raw=False, ignore_index=None, mask=None):
    """Calculate per class pixel accuracy.

    Args:
        pred (ndarray): Prediction segmentation map (H, W).
        label (ndarray): Ground truth segmentation map (H, W).

    Returns:
        ndarray: Per class pixel accuracy (num_classes,).
    """
    # from sklearn.metrics import confusion_matrix
    # matrix = confusion_matrix(pred, label)
    # return matrix.diagonal()/matrix.sum(axis=1)
    if ignore_index:
        index_mask = torch.logical_and(ignore_indices(label, ignore_index), ignore_indices(pred, ignore_index))
        if mask is not None:
            mask = torch.logical_and(mask, index_mask)
        else:
            mask = index_mask

    matrix = confusion_matrix(pred, label, 19, is_torch=(isinstance(pred, torch.Tensor)), mask=mask)
    return matrix if return_raw else matrix.diagonal()/matrix.sum(axis=1)

def confusion_matrix(pred, label, num_classes, is_torch=False, mask=None):
    """Calculate confusion matrix.

    Args:
        pred (tensor): Prediction segmentation map (H, W).
        label (tensor): Ground truth segmentation map (H, W).

    Returns:
        ndarray: Confusion matrix (num_classes, num_classes).
    """
    # assert len(pred.shape) == 2, f"pred has wrong dimension.  Got{pred.shape}"
    # assert len(label.shape) == 2, f"label has wrong dimension.  Got{label.shape}"
    # num_classes = max(pred.max(), label.max()) + 1
    if mask is None:
        mask = (label >= 0) & (label < num_classes) & (pred >= 0) & (pred < num_classes)
    if is_torch:
        label = num_classes * label[mask].long() + pred[mask].long()
        count = torch.bincount(label, minlength=num_classes**2)
    else:
        label = num_classes * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, ax, class_names=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    confusion_matrix: (num_classes, num_classes)
    plots a visualization of the confusion matrix on the given axis
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(confusion_matrix)

    ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(confusion_matrix.shape[0]))
    ax.set_yticks(np.arange(confusion_matrix.shape[1]))
    if class_names is not None:
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    ax.set_ylim(confusion_matrix.shape[0] - 0.5, -0.5)
    ax.set_xlim(-0.5, confusion_matrix.shape[1] - 0.5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_aspect('equal')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, f"{confusion_matrix[i, j]:.2f}", ha="center", va="center", color="w")
    return ax


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
