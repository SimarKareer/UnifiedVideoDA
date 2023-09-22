# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix, intersect_and_union, confusion_matrix, plot_confusion_matrix, per_class_pixel_accuracy
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, one_mix, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg, get_segmentation_error_vis
from mmseg.ops import resize
from mmseg.utils.utils import downscale_label_ratio
from mmseg.utils.custom_utils import three_channel_flow
from mmseg.datasets.cityscapes import CityscapesDataset
from tools.aggregate_flows.flow.my_utils import errorVizClasses, multiBarChart, backpropFlow, backpropFlowNoDup, visFlow, tensor_map, rare_class_or_filter, invNorm, CircularTensor
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
from mmseg.models.segmentors.accel_hrda_encoder_decoder import ACCELHRDAEncoderDecoder

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.debug_mode = cfg["debug_mode"]
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.source_only2 = cfg['source_only2']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.l_warp_lambda = cfg['l_warp_lambda']
        self.l_mix_lambda = cfg['l_mix_lambda']
        self.consis_filter = cfg['consis_filter']
        self.consis_confidence_filter = cfg['consis_confidence_filter']
        self.consis_confidence_thresh = cfg['consis_confidence_thresh']
        self.consis_confidence_per_class_thresh = cfg['consis_confidence_per_class_thresh']
        self.consis_filter_rare_class = cfg["consis_filter_rare_class"]
        self.oracle_mask_add_noise = cfg['oracle_mask_add_noise']
        self.oracle_mask_remove_pix = cfg['oracle_mask_remove_pix']
        self.oracle_mask_noise_percent = cfg['oracle_mask_noise_percent']
        self.TPS_warp_pl_confidence = cfg['TPS_warp_pl_confidence']
        self.TPS_warp_pl_confidence_thresh = cfg['TPS_warp_pl_confidence_thresh']
        self.max_confidence = cfg["max_confidence"]

        self.pl_fill = cfg['pl_fill']
        self.bottom_pl_fill = cfg['bottom_pl_fill']
        self.oracle_mask = cfg['oracle_mask']
        self.warp_cutmix = cfg['warp_cutmix']
        self.exclusive_warp_cutmix = cfg['exclusive_warp_cutmix']
        self.stub_training = cfg['stub_training']
        self.l_warp_begin = cfg['l_warp_begin']
        self.class_mask_warp = cfg["class_mask_warp"]
        self.class_mask_cutmix = cfg["class_mask_cutmix"]
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']

        self.multimodal = cfg["modality"] != "rgb"
        self.modality = cfg["modality"]
        self.modality_dropout_weights = cfg["modality_dropout_weights"]

        self.ignore_index = cfg["ignore_index"]
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        if self.class_mask_warp == "thing":
            self.relevant_classes_warp = list(CityscapesDataset.THINGS_IDX)
            self.masked_classes_warp = list(CityscapesDataset.STUFF_IDX)
        elif self.class_mask_warp == "stuff":
            self.relevant_classes_warp = list(CityscapesDataset.STUFF_IDX)
            self.masked_classes_warp = list(CityscapesDataset.THINGS_IDX)
        else:
            self.relevant_classes_warp = list(CityscapesDataset.ALL_IDX)
            self.masked_classes_warp = []

        if self.class_mask_cutmix == "thing":
            self.relevant_classes_cutmix = list(CityscapesDataset.THINGS_IDX)
            self.masked_classes_cutmix = list(CityscapesDataset.STUFF_IDX)
        elif self.class_mask_cutmix == "stuff":
            self.relevant_classes_cutmix = list(CityscapesDataset.STUFF_IDX)
            self.masked_classes_cutmix = list(CityscapesDataset.THINGS_IDX)
        else:
            self.relevant_classes_cutmix = list(CityscapesDataset.ALL_IDX)
            self.masked_classes_cutmix = []


        self.per_class_confidence_thresh = None
        if self.consis_confidence_per_class_thresh:
            self.per_class_confidence_thresh = {i: CircularTensor(1000) for i in range(self.num_classes)}

        self.init_cml_debug_metrics()
        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.accel = isinstance(self.model, ACCELHRDAEncoderDecoder)

    def init_cml_debug_metrics(self):
        
        self.cml_debug_metrics = {k: torch.zeros(19) for k in [
            "warp_cml_intersect", "plain_cml_intersect", "plain_mask_cml_intersect", 
            "warp_cml_pixel_hit", "plain_cml_pixel_hit", "plain_mask_cml_pixel_hit", 
            "warp_cml_union", "plain_cml_union", "plain_mask_cml_union", 
            "warp_cml_pixel_total", "plain_cml_pixel_total", "plain_mask_cml_pixel_total",
            "warp_cml_mask_hit", 
            "warp_cml_mask_total"]}  

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight
    
    def save_image_grid(self, images, path):
        grid = torchvision.utils.make_grid(images, nrow=4)
        # grid = grid.permute(1, 2, 0)
        grid = grid.cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        cv2.imwrite(path, grid)

    def get_pl(self, target_img, target_img_metas, seg_debug, seg_debug_key, valid_pseudo_mask, return_logits=False, flow=None):
        ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas, flow=flow)
        
        if seg_debug:
            seg_debug[seg_debug_key] = self.get_ema_model().debug_output

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
            ema_logits)

        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)

        if return_logits:
            ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
            return pseudo_label, pseudo_weight, ema_softmax
        else:
            del ema_logits
            return pseudo_label, pseudo_weight
    
    def fast_iou(self, pred, label, custom_mask=None, return_raw=False):
        pl_warp_intersect, pl_warp_union, _, _, mask = intersect_and_union(
            pred,
            label,
            19,
            # [5, 3, 16, 12, 201, 255],
            self.ignore_index + self.masked_classes_warp,
            label_map=None,
            reduce_zero_label=False,
            custom_mask = custom_mask,
            return_mask=True
        )
        iou = (pl_warp_intersect / pl_warp_union).numpy()
        miou = np.nanmean(iou)
        if return_raw:
            return pl_warp_intersect, pl_warp_union, mask
        else:
            return iou, miou, mask

    def get_mixed_im(self, pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds, img_flow, target_img_flow):
        B = img.shape[0]
        if self.accel:
            mixed_img, mixed_lbl, mixed_seg_weight, mix_masks = self.get_mixed_im_helper(pseudo_weight, pseudo_label, img[:B//2], target_img[:B//2], gt_semantic_seg, means, stds)
            # is mix masks 1 for source or target?  1 is source, 0 target

            mixed_flow = target_img_flow.clone()
            mix_masks_repeat = (torch.cat(mix_masks, dim=0) == 1).repeat((1, 2, 1, 1))
            mixed_flow[mix_masks_repeat] = img_flow[mix_masks_repeat]
            mixed_img2, mixed_lbl2, mixed_seg_weight2, mix_masks2 = self.get_mixed_im_helper(pseudo_weight, pseudo_label, img[B//2:], target_img[B//2:], gt_semantic_seg, means, stds, mix_masks=mix_masks)

            # just return the mixed_img, mixed_lbl, mixed_seg_weight, mix_masks after stacking mixed_img2.
            return torch.cat((mixed_img, mixed_img2), dim=0), mixed_lbl, mixed_seg_weight, mix_masks, mixed_flow
        else:
            return self.get_mixed_im_helper(pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds)

    def get_mixed_im_helper(self, pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds, mix_masks=None):
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        # if self.accel:
        #     assert img.shape[0] == target_img.shape[0] == pseudo_label.shape[0] * 2
        # else:
        #     assert img.shape[0] == pseudo_weight.shape[0] == pseudo_label.shape[0]
        batch_size = img.shape[0]
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=img.device)

        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        class_filter = self.masked_classes_cutmix

        mix_masks = get_class_masks(gt_semantic_seg, class_filter=class_filter) if mix_masks is None else mix_masks

        # if self.accel:
        #     # NOTE: this enables consistent cutmix
        #     mix_masks = mix_masks + mix_masks

        # fig, axs = plt.subplots(5, 8, figsize=(15, 15))
        # subplotimg(axs[0, 0], img[:, 5], "Img before", cmap="Greys")
        # subplotimg(axs[0, 1], target_img[:, 5], "target_img before", cmap="Greys")
        # subplotimg(axs[1, 0], invNorm(img[0, :3]), "Img Before")
        # subplotimg(axs[1, 1], invNorm(target_img[0, :3]), "Target Img Before")


        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack(
                    (gt_semantic_seg[i][0], pseudo_label[i])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        del gt_pixel_weight
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # subplotimg(axs[0, 2], mixed_img[:, 5], "Mixed after", cmap="Greys")
        # subplotimg(axs[1, 2], invNorm(mixed_img[0, :3]), "Mixed after")
        # fig.savefig("work_dirs/debug/transform.png")
        # breakpoint()


        return mixed_img, mixed_lbl, mixed_seg_weight, mix_masks
    
    def get_mixed_gt(self, mix_masks, gt_semantic_seg, target_img_extra):
        batch_size = gt_semantic_seg.shape[0]
        mixed_gt = [None] * batch_size
        for i in range(batch_size):
            _, mixed_gt[i] = one_mix(mask=mix_masks[i], 
                                data=None, 
                                target=torch.stack((gt_semantic_seg[i][0], target_img_extra["gt_semantic_seg"][i][0])))
        return mixed_gt

    def add_training_debug_metrics(self, pseudo_label, pseudo_label_warped, pseudo_weight_warped, target_img_extra, log_vars):
        pl_warp = pseudo_label_warped[0]
        pw_warp = pseudo_weight_warped[0]
        gt_sem_seg = target_img_extra["gt_semantic_seg"][0, 0]

        # All IoU Metrics
        warp_iou = self.fast_iou(pl_warp.cpu().numpy(), gt_sem_seg.cpu().numpy(), custom_mask=pw_warp.cpu().bool().numpy(), return_raw=True)
        self.cml_debug_metrics["warp_cml_intersect"] += warp_iou[0]
        self.cml_debug_metrics["warp_cml_union"] += warp_iou[1]

        plain_iou_mask = self.fast_iou(pseudo_label[0].cpu().numpy(), gt_sem_seg.cpu().numpy(), custom_mask=pw_warp.cpu().bool().numpy(), return_raw=True)
        self.cml_debug_metrics["plain_mask_cml_intersect"] += plain_iou_mask[0]
        self.cml_debug_metrics["plain_mask_cml_union"] += plain_iou_mask[1]

        plain_iou = self.fast_iou(pseudo_label[0].cpu().numpy(), gt_sem_seg.cpu().numpy(), return_raw=True)
        self.cml_debug_metrics["plain_cml_intersect"] += plain_iou[0]
        self.cml_debug_metrics["plain_cml_union"] += plain_iou[1]

        # All Pixel Accuracy Metrics
        warp_pixel_acc = per_class_pixel_accuracy(pl_warp, gt_sem_seg, ignore_index=self.ignore_index + self.masked_classes_warp, return_raw=True).cpu()
        self.cml_debug_metrics["warp_cml_pixel_hit"] += warp_pixel_acc.diagonal()
        self.cml_debug_metrics["warp_cml_pixel_total"] += warp_pixel_acc.sum(axis=1)

        plain_pixel_acc = per_class_pixel_accuracy(pseudo_label[0], gt_sem_seg, ignore_index=self.ignore_index + self.masked_classes_warp, return_raw=True).cpu()
        self.cml_debug_metrics["plain_cml_pixel_hit"] += plain_pixel_acc.diagonal()
        self.cml_debug_metrics["plain_cml_pixel_total"] += plain_pixel_acc.sum(axis=1)

        plain_mask_pixel_acc = per_class_pixel_accuracy(pseudo_label[0], gt_sem_seg, ignore_index=self.ignore_index + self.masked_classes_warp, return_raw=True, mask=pw_warp.bool()).cpu()
        self.cml_debug_metrics["plain_mask_cml_pixel_hit"] += plain_mask_pixel_acc.diagonal()
        self.cml_debug_metrics["plain_mask_cml_pixel_total"] += plain_mask_pixel_acc.sum(axis=1)

        # Mask Counts Per Class
        # breakpoint()
        total_counts = pseudo_label[0][pseudo_label[0] != 255].unique(return_counts=True)
        mask_counts = pseudo_label[0][(pseudo_label[0] != 255) & ~pw_warp.bool()].unique(return_counts=True)
        self.cml_debug_metrics["warp_cml_mask_total"][total_counts[0].cpu()] += total_counts[1].cpu()
        self.cml_debug_metrics["warp_cml_mask_hit"][mask_counts[0].cpu()] += mask_counts[1].cpu()

        for i, class_name in enumerate(CityscapesDataset.CLASSES):
            log_vars[f"Warp PL IoU {class_name}"] = self.cml_debug_metrics["warp_cml_intersect"][i] / self.cml_debug_metrics["warp_cml_union"][i]
            log_vars[f"Plain PL IoU {class_name}"] = self.cml_debug_metrics["plain_cml_intersect"][i] / self.cml_debug_metrics["plain_cml_union"][i]
            # log_vars[f"Plain PL Mask IoU {class_name}"] = self.cml_debug_metrics["plain_mask_cml_intersect"][i] / self.cml_debug_metrics["plain_mask_cml_union"][i]

            log_vars["Warp PL mIoU"] = np.nanmean((self.cml_debug_metrics["warp_cml_intersect"] / self.cml_debug_metrics["warp_cml_union"]).numpy())
            log_vars["Plain PL mIoU"] = np.nanmean((self.cml_debug_metrics["plain_cml_intersect"] / self.cml_debug_metrics["plain_cml_union"]).numpy())
            # log_vars["Plain PL Mask mIoU"] = np.nanmean((self.cml_debug_metrics["plain_mask_cml_intersect"] / self.cml_debug_metrics["plain_mask_cml_union"]).numpy())

            log_vars[f"Warp PL Pixel Acc {class_name}"] = self.cml_debug_metrics["warp_cml_pixel_hit"][i] / self.cml_debug_metrics["warp_cml_pixel_total"][i]
            log_vars[f"Plain PL Pixel Acc {class_name}"] = self.cml_debug_metrics["plain_cml_pixel_hit"][i] / self.cml_debug_metrics["plain_cml_pixel_total"][i]
            # log_vars[f"Plain PL Mask Pixel Acc {class_name}"] = self.cml_debug_metrics["plain_mask_cml_pixel_hit"][i] / self.cml_debug_metrics["plain_mask_cml_pixel_total"][i]

            log_vars[f"Diff Pixel Acc {class_name}"] = log_vars[f"Warp PL Pixel Acc {class_name}"] - log_vars[f"Plain PL Pixel Acc {class_name}"]
            # log_vars[f"Diff Mask Pixel Acc {class_name}"] = log_vars[f"Warp PL Pixel Acc {class_name}"] - log_vars[f"Plain PL Mask Pixel Acc {class_name}"]
            log_vars[f"Diff IoU {class_name}"] = log_vars[f"Warp PL IoU {class_name}"] - log_vars[f"Plain PL IoU {class_name}"]
            # log_vars[f"Diff Mask IoU {class_name}"] = log_vars[f"Warp PL IoU {class_name}"] - log_vars[f"Plain PL Mask IoU {class_name}"]
            log_vars["Diff mIoU"] = log_vars["Warp PL mIoU"] - log_vars["Plain PL mIoU"]
            # log_vars["Diff Mask mIoU"] = log_vars["Warp PL mIoU"] - log_vars["Plain PL Mask mIoU"]

            log_vars[f"Mask Percentage {class_name}"] = self.cml_debug_metrics["warp_cml_mask_hit"][i] / self.cml_debug_metrics["warp_cml_mask_total"][i]
    
    def add_iou_metrics(self, pred, gt_sem_seg, log_vars, name):
        """
        pseudo_label: B, H, W
        gt_sem_seg: B, 1, H, W
        log_vars: {varName:v}
        """
        gt_sem_seg = gt_sem_seg[0, 0]
        plain_iou = self.fast_iou(pred[0].cpu().numpy(), gt_sem_seg.cpu().numpy(), return_raw=True)
        self.cml_debug_metrics["plain_cml_intersect"] += plain_iou[0]
        self.cml_debug_metrics["plain_cml_union"] += plain_iou[1]

        for i, class_name in enumerate(CityscapesDataset.CLASSES):
            log_vars[f"Plain {name} IoU {class_name}"] = self.cml_debug_metrics["plain_cml_intersect"][i] / self.cml_debug_metrics["plain_cml_union"][i]

        log_vars[f"Plain {name} mIoU"] = np.nanmean((self.cml_debug_metrics["plain_cml_intersect"] / self.cml_debug_metrics["plain_cml_union"]).numpy())
        
        return log_vars

    def get_grad_magnitude(self):
        params = self.get_model().backbone.parameters()
        seg_grads = [
            p.grad.detach().clone() for p in params if p.grad is not None
        ]
        grad_mag = calc_grad_magnitude(seg_grads)
        return grad_mag
    
    def imnet_feat_dist(self, img, gt_semantic_seg, src_feat, log_vars, log_prefix = None):
        # ImageNet feature distance
        # if self.enable_fdist:
        feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                    src_feat)
        feat_log = add_prefix(feat_log, 'src')
        if log_prefix is not None:
            feat_log = add_prefix(feat_log, log_prefix)
        log_vars.update(feat_log)
        return feat_loss

    def forward_train_multimodal(self, img, img_metas, img_extra, target_img, target_img_metas, target_img_extra):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        DEBUG = self.debug_mode


        # concat flow and img.  NOTE, using a copied version.
        assert(len(img_extra["flowVis"].shape) == 4)
        img = torch.cat([img, three_channel_flow(img_extra["flowVis"].clone())], dim=1)
        target_img = torch.cat([target_img, three_channel_flow(target_img_extra["flowVis"].clone())], dim=1)

        # Assign other important variables used throughout function.  Try not to edit the dictionary anywhere
        gt_semantic_seg, valid_pseudo_mask = img_extra["gt_semantic_seg"], target_img_extra["valid_pseudo_mask"]

        if DEBUG:
            fig, axs = plt.subplots(5, 8, figsize=(15, 15))
            subplotimg(axs[0, 0], invNorm(img[0][:3]), "Source IM 0")
            # subplotimg(axs[0, 1], invNorm(img[1][:3]), "Source IM 1")
            subplotimg(axs[0, 2], invNorm(target_img[0][:3]), "Target IM 0")
            # subplotimg(axs[0, 3], invNorm(target_img[1][:3]), "Target IM 1")

            subplotimg(axs[1, 0], img_extra["flowVis"][0], "Source flow 0")
            # subplotimg(axs[1, 1], img_extra["flowVis"][1], "Source flow 1")
            subplotimg(axs[1, 2], target_img_extra["flowVis"][0], "Target flow 0")
            # subplotimg(axs[1, 3], target_img_extra["flowVis"][1], "Target flow 1")

        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)
        
        self.update_debug_state()
        seg_debug = {}
        means, stds = get_mean_std(img_metas, dev)

        # Train on source images
        clean_losses = self.model(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.source_only2:
            del src_feat, clean_loss
            del seg_debug
            return log_vars

        if self.print_grad_magnitude:
            mmcv.print_log(f'Seg. Grad.: {self.get_grad_magnitude()}', 'mmseg')

        
        # if self.enable_fdist:
        #     breakpoint()
        #     feat_loss_rgb = self.imnet_feat_dist(img, gt_semantic_seg, src_feat[0], log_vars, log_prefix='rgb')
        #     feat_loss_rgb.backward(retain_graph=True)
        #     feat_loss_flow = self.imnet_feat_dist(img, gt_semantic_seg, src_feat[1], log_vars, log_prefix='flow')
        #     feat_loss_flow.backward()
        #     del src_feat, clean_loss
        #     del feat_loss_rgb, feat_loss_flow

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        pseudo_label, pseudo_weight = self.get_pl(target_img, target_img_metas, seg_debug, "Target", valid_pseudo_mask)
        self.add_iou_metrics(pseudo_label, target_img_extra["gt_semantic_seg"], log_vars, name="PL Target-only")
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        log_vars["L_warp"] = 0

        if self.oracle_mask_add_noise or self.oracle_mask_remove_pix:
            # oracle masking
            oracle_map = (pseudo_label == target_img_extra["gt_semantic_seg"].squeeze(1)) & (target_img_extra["gt_semantic_seg"].squeeze(1) != 255)
            pseudo_label_oracle_consis = torch.clone(pseudo_label)
            pseudo_label_oracle_consis_weight = torch.clone(pseudo_weight)

            pseudo_label_oracle_consis_weight[~oracle_map] = 0
            pseudo_label_oracle_consis[~oracle_map] = 255

            inconsis_pixels = (pseudo_label != target_img_extra["gt_semantic_seg"].squeeze(1)) & (target_img_extra["gt_semantic_seg"].squeeze(1) != 255)

            for i in range(batch_size):
                

                if self.oracle_mask_add_noise:
                    inconsis_pix_i = inconsis_pixels[i]
                    # print("Before non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())
                    # print(self.oracle_mask_noise_percent)

                    num_inconsis = inconsis_pix_i.sum().item()
                    num_noise = min(int(self.oracle_mask_noise_percent * oracle_map[i].sum().item()), num_inconsis)

                    num_to_change = num_inconsis - num_noise

                    true_indices = torch.where(inconsis_pix_i)

                    change_indices = np.random.choice(len(true_indices[0]), size=num_to_change, replace=False)
                
                    inconsis_pix_i[true_indices[0][change_indices], true_indices[1][change_indices]] = False

                    # print("after", inconsis_pix_i.sum().item())

                    pseudo_label_oracle_consis[i][inconsis_pix_i] = pseudo_label[i][inconsis_pix_i]
                    pseudo_label_oracle_consis_weight[i][inconsis_pix_i] = pseudo_weight[i][inconsis_pix_i]

                    # print("after non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())

                if self.oracle_mask_remove_pix:
                    # print("Before non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())
                    num_to_change = int(self.oracle_mask_noise_percent * oracle_map[i].sum().item())

                    true_indices = torch.where(oracle_map[i])

                    change_indices = np.random.choice(len(true_indices[0]), size=num_to_change, replace=False)

                    oracle_map[i][true_indices[0][change_indices], true_indices[1][change_indices]] = False

                    pseudo_label_oracle_consis[i][~oracle_map[i]] = 255
                    pseudo_label_oracle_consis_weight[i][~oracle_map[i]] = 0

                    # print("after non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())

            pseudo_label = pseudo_label_oracle_consis
            pseudo_weight = pseudo_label_oracle_consis_weight

        if self.l_mix_lambda > 0:
            # Apply mixing
            mixed_img, mixed_lbl, mixed_seg_weight, mix_masks = self.get_mixed_im(pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds)
            if DEBUG:
                subplotimg(axs[2, 3], invNorm(mixed_img[0][:3]), "Mixed Im with CutMix")
                subplotimg(axs[2, 4], mixed_img[0][None, 3], "Mixed flow with CutMix")
                subplotimg(axs[2, 5], mixed_lbl[0], "Mixed lbl with CutMix", cmap="cityscapes")
                subplotimg(axs[2, 6], mixed_seg_weight[0].repeat(3, 1, 1)*255)
            # Train on mixed images
            if self.modality_dropout_weights is not None:
                masking_branch = random.choices([0, 1, -1], weights = self.modality_dropout_weights, k = mixed_img.shape[0])
            else:
                masking_branch = None
            mix_losses = self.model(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
                return_logits=True,
                masking_branch=masking_branch
            )
            if isinstance(mix_losses['decode.logits'], tuple):
                mix_student_pred = torch.softmax(mix_losses['decode.logits'][0].detach(), dim=1) #indexing 0th to get the fused predictions - which is what is used in decode_head.forward_test
            else:
                mix_student_pred = torch.softmax(mix_losses['decode.logits'].detach(), dim=1) #For non HRDA don't index into tuple
            mixed_gt = torch.cat(self.get_mixed_gt(mix_masks, gt_semantic_seg, target_img_extra))
            mix_student_pred = resize(
                input=mix_student_pred,
                size=mixed_gt.shape[2:],
                mode='bilinear',
                align_corners=self.get_model().align_corners)
            _, mix_student_pred = torch.max(mix_student_pred, dim=1)
            self.add_iou_metrics(mix_student_pred, mixed_gt, log_vars, name="Mixed Student")
            del mix_losses['decode.logits'], mix_student_pred, mixed_gt

            seg_debug['Mix'] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            (mix_loss * self.l_mix_lambda).backward()
        
        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.model, img, img_metas,
                                gt_semantic_seg, target_img,
                                target_img_metas, valid_pseudo_mask,
                                pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        if DEBUG:
            subplotimg(axs[2, 0], pseudo_label, "PL", cmap="cityscapes")
            fig.savefig(f"./work_dirs/multimodal/debug{self.local_iter}.png")

        
        if self.local_iter % 100 == 0:
            self.init_cml_debug_metrics()
        self.local_iter += 1
        return log_vars


    def accel_format(self, img, img_metas, img_extra):
        """
        Use img_extra to stack imt and imt-1, imt-1, imt-2 in the batch dimension
        """
        img_extra["img"] = torch.cat((img_extra["img[0]"], img_extra["img[-1]"]), dim=0)
        img = img_extra["img"]
        img_extra["imtk"] = torch.cat((img_extra["img[-1]"], img_extra["img[-2]"]), dim=0)

        # if img_extra["gt_semantic_seg[-1]"].numel() != 0:
        #     img_extra["gt_semantic_seg"] = torch.cat((img_extra["gt_semantic_seg[0]"], img_extra["gt_semantic_seg[-1]"]), dim=0)

        if "valid_pseudo_mask[0]" in img_extra:
            img_extra["valid_pseudo_mask"] = img_extra.pop("valid_pseudo_mask[0]")

        return img, img_metas, img_extra

    def legacy_format(self, img_extra, target_img_extra):
            # rename img_extras for this test
            img_extra["imtk"] = img_extra.pop("img[-1]")
            img_extra["imtk_gt_semantic_seg"] = img_extra.pop("gt_semantic_seg[-1]")
            img_extra["imtk_flow"] = img_extra.pop("flow[-1]")
            # img_extra["imtk_metas"] = img_extra.pop("img_metas[-1]")
            img_extra["img"] = img_extra.pop("img[0]")
            img_extra["gt_semantic_seg"] = img_extra.pop("gt_semantic_seg[0]")
            img_extra["flow"] = img_extra.pop("flow[0]")
            img_extra["img_metas"] = img_extra.pop("img_metas")

            target_img_extra["imtk"] = target_img_extra.pop("img[-1]")
            target_img_extra["imtk_gt_semantic_seg"] = target_img_extra.pop("gt_semantic_seg[-1]")
            target_img_extra["imtk_flow"] = target_img_extra.pop("flow[-1]")
            # target_img_extra["imtk_metas"] = target_img_extra.pop("img_metas[-1]")
            target_img_extra["img"] = target_img_extra.pop("img[0]")
            target_img_extra["gt_semantic_seg"] = target_img_extra.pop("gt_semantic_seg[0]")
            target_img_extra["flow"] = target_img_extra.pop("flow[0]")
            target_img_extra["img_metas"] = target_img_extra.pop("img_metas")
            target_img_extra["valid_pseudo_mask"] = target_img_extra.pop("valid_pseudo_mask[0]")
            # breakpoint()


    def forward_train(self, img, img_metas, img_extra, target_img, target_img_metas, target_img_extra):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.accel: # ACCEL
            img, img_metas, img_extra = self.accel_format(img, img_metas, img_extra)
            target_img, target_img_metas, target_img_extra = self.accel_format(target_img, target_img_metas, target_img_extra)
        else: # LEGACY
            self.legacy_format(img_extra, target_img_extra)

        if self.multimodal:
            return self.forward_train_multimodal(img, img_metas, img_extra, target_img, target_img_metas, target_img_extra)

        gt_semantic_seg, valid_pseudo_mask, = img_extra["gt_semantic_seg"], target_img_extra["valid_pseudo_mask"]
        
        log_vars = {}
        if self.stub_training:
            return log_vars
        batch_size = img.shape[0] // 2 if self.accel else img.shape[0]
        H, W = target_img_extra["imtk"].shape[2:]
        B = batch_size
        dev = img.device

        DEBUG = self.debug_mode

        if DEBUG:
            rows, cols = 7, 7
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(3 * cols, 3 * rows),
            )

            large_fig, large_axs = plt.subplots(
                2,
                2,
                figsize=(8 * cols, 8 * rows),
            )

            subplotimg(axs[0, 0], invNorm(img[0][:3]), "Source IM 0")
            subplotimg(axs[0, 1], invNorm(img_extra["imtk"][0][:3]), "Source IM 1")
            # subplotimg(axs[0, 1], invNorm(img[1][:3]), "Source IM 1")
            subplotimg(axs[0, 2], invNorm(target_img[0][:3]), "Target IM 0")
            subplotimg(axs[0, 3], invNorm(target_img_extra["imtk"][0][:3]), "Target IM 1")
            # subplotimg(axs[0, 3], invNorm(target_img[1][:3]), "Target IM 1")

            # subplotimg(axs[1, 0], img_extra["flowVis"][0], "Source flow 0")
            # subplotimg(axs[1, 1], img_extra["flowVis"][1], "Source flow 1")
            # subplotimg(axs[1, 2], target_img_extra["flowVis"][0], "Target flow 0")
            # subplotimg(axs[1, 3], target_img_extra["flowVis"][1], "Target flow 1")

        # if self.local_iter % 5 == 0:
        #     for i in range(torch.cuda.device_count()):
        #         print(torch.cuda.memory_summary(i))
        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}
        means, stds = get_mean_std(img_metas, dev)

        # Train on source images
        clean_losses = self.model(img, img_metas, gt_semantic_seg, flow=img_extra["flow[0]"], return_feat=True)
        # clean_losses = self.model(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)

        if self.source_only2:
            del src_feat, clean_loss
            del seg_debug

            if DEBUG:
                fig.savefig(f"work_dirs/LWarpPLAnalysis/ims{self.local_iter}.png", dpi=200)
                large_fig.savefig(f"work_dirs/LWarpPLAnalysis/graphs{self.local_iter}.png", dpi=200)
            return log_vars
        
        if self.print_grad_magnitude:
            mmcv.print_log(f'Seg. Grad.: {self.get_grad_magnitude()}', 'mmseg')
            

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img[:img.shape[0]//2], gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        need_logits = self.consis_confidence_filter or self.TPS_warp_pl_confidence or self.max_confidence
        if not self.source_only:
            target_img_fut, target_img_fut_metas = target_img_extra["imtk"], target_img_extra["img_metas"]

            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            
            if need_logits:
                pseudo_label, pseudo_weight, logits_curr = self.get_pl(target_img, target_img_metas, seg_debug, "Target", valid_pseudo_mask, return_logits=True, flow=target_img_extra["flow[0]"])
                pseudo_label_fut, pseudo_weight_fut, logits_fut = self.get_pl(target_img_fut, target_img_fut_metas, None, None, valid_pseudo_mask, return_logits=True, flow=target_img_extra["flow[-1]"]) #This mask isn't dynamic so it's fine to use same for pl and pl_fut
            else:
                pseudo_label, pseudo_weight = self.get_pl(target_img, target_img_metas, seg_debug, "Target", valid_pseudo_mask, flow=target_img_extra["flow[0]"])
                pseudo_label_fut, pseudo_weight_fut = self.get_pl(target_img_fut, target_img_fut_metas, None, None, valid_pseudo_mask, flow=target_img_extra["flow[-1]"]) #This mask isn't dynamic so it's fine to use same for pl and pl_fut
            
            self.add_iou_metrics(pseudo_label, target_img_extra["gt_semantic_seg"], log_vars, name="PL Target-only")
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            log_vars["L_warp"] = 0
            if DEBUG or self.l_warp_lambda > 0 and self.local_iter >= self.l_warp_begin: 
                pseudo_label_warped = [] #pseudo_label_fut.clone() #Note: technically don't need to clone, could be faster
                pseudo_weight_warped = []
                masks = []
                for i in range(batch_size):
                    flowi = target_img_extra["flow"][i].permute(1, 2, 0)
                    pli = pseudo_label_fut[[i]].permute(1, 2, 0)

                    # So technically this was unnecessary bc the pseudo_weight is always just 1 number
                    if need_logits:
                        pli_and_weight = torch.cat((pli, pseudo_weight_fut[[i]].permute(1, 2, 0), logits_fut[[i]][0].permute(1,2,0)), dim=2)
                    else:
                        pli_and_weight = torch.cat((pli, pseudo_weight_fut[[i]].permute(1, 2, 0)), dim=2)
                    warped_stack, mask = backpropFlow(flowi, pli_and_weight, return_mask=True, return_mask_count=False) #TODO, will need to stack pli with weights if we want warped weights
                    if DEBUG and i == 0:
                        masks.append(mask.cpu().numpy())
                        subplotimg(axs[3, 0], mask.repeat(3, 1, 1)*255, "Warping Mask")
                    pltki, pseudo_weight_warped_i = warped_stack[:, :, [0]], warped_stack[:, :, 1]

                    if need_logits:
                        logits_warped_i = warped_stack[:, :, 2:]
                        logits_warped_i = logits_warped_i.permute((2, 0, 1)).float()
                    
                    pseudo_weight_warped_i = pseudo_weight_warped_i.float()
                    pseudo_weight_warped_i[~mask] = 0
                    pltki = pltki.permute((2, 0, 1)).long()

                    # PL Fill in
                    if self.pl_fill or self.bottom_pl_fill:
                        if i == 0 and DEBUG:
                            subplotimg(axs[5][2], pseudo_weight_warped_i.repeat(3, 1, 1)*255, 'PW Before Fill')
                            subplotimg(axs[5][0], pltki[0], 'PL Before Fill', cmap="cityscapes")

                        if self.pl_fill:
                            pseudo_weight_warped_i[~mask] = pseudo_weight[i][~mask]
                            pltki[0][~mask] = pseudo_label[i][~mask]
                        
                        if self.bottom_pl_fill:
                            # only set the bottom of the pseudo_weight_warped_i to the pseudo_weight
                            # breakpoint()
                            H, W = pseudo_weight_warped_i.shape
                            bottom_mask = torch.zeros_like(mask)
                            bottom_mask[800:, :] = 1
                            bottom_mask[:, :100] = 1
                            bottom_mask[:, W-100:] = 1
                            if DEBUG:
                                subplotimg(axs[2, 4], bottom_mask.repeat(3, 1, 1)*255, "corners mask")
                            bottom_mask = bottom_mask & (pseudo_weight_warped_i == 0)
                            # bottom_mask = (torch.ones_like(mask) & torch.arange(mask.shape[0])[:, None].cuda() >= 800) | (torch.arange(mask.shape[0])[None, :].cuda() <= 100 | torch.arange(mask.shape[0])[None, :].cuda() >= W-100) & (pseudo_weight_warped_i != 0) # fills in bottom and sides
                            pseudo_weight_warped_i[bottom_mask] = pseudo_weight[i][bottom_mask]
                            pltki[0][bottom_mask] = pseudo_label[i][bottom_mask]

                        if i == 0 and DEBUG:
                            subplotimg(axs[5][3], pseudo_weight_warped_i.repeat(3, 1, 1)*255, 'PW After Fill')
                            subplotimg(axs[5][4], pseudo_weight[i].repeat(3, 1, 1)*255, 'PW Original')
                            subplotimg(axs[5][1], pltki[0], 'PL After Fill', cmap="cityscapes")
                    

                    # Consistency masking
                    if self.consis_filter:
                        pseudo_weight_warped_i[pltki[0] != pseudo_label[i]] = 0
                        pltki[0][pltki[0] != pseudo_label[i]] = 255
                        if i == 0 and DEBUG:
                            subplotimg(axs[3][5], (pltki[0] != pseudo_label[i]).repeat(3, 1, 1) * 255, 'Consistency Map')

                    if self.max_confidence:
                        logits_curr_max_i = logits_curr.max(dim=1).values[i]
                        logits_warped_max_i = logits_warped_i.max(dim=0).values
                        curr_larger = (logits_curr_max_i > logits_warped_max_i) | (logits_warped_max_i == 255)
                        pltki[0][curr_larger] = pseudo_label[i][curr_larger]
                        pseudo_weight_warped_i[curr_larger] = pseudo_weight[i][curr_larger]

                    if self.TPS_warp_pl_confidence:
                        # logits for predicted values
                        max_logit_val_fut , max_logit_idx_fut = torch.max(logits_warped_i, dim=0)

                        # only take the warped pixels which have a confidence > threshold
                        pltki[0][max_logit_val_fut < self.TPS_warp_pl_confidence_thresh] = 255
                        pseudo_weight_warped_i[max_logit_val_fut < self.TPS_warp_pl_confidence_thresh] = 0
                        if i == 0 and DEBUG:
                            subplotimg(axs[6][3], pltki[0], "TPS tk confidence",  cmap="cityscapes")
                        
                    if self.consis_confidence_filter:
                        logits_curr_i = logits_curr[[i]][0]                        
                        pseudo_label_i = pseudo_label[i]


                        #consis mask
                        consis_mask = (pltki[0] == pseudo_label_i)

                        # logits for predicted values

                        max_logit_val_fut , max_logit_idx_fut = torch.max(logits_warped_i, dim=0)
                        max_logit_val_curr ,max_logit_idx_curr = torch.max(logits_curr_i, dim=0)

                        # confidence mask above threshold
                        # print("Num pixles in PL consis", consis_mask.sum())

                        if self.consis_confidence_per_class_thresh:
                            # cummulative mask from all classes
                            confidence_mask = None

                            for i in range(self.num_classes):

                                #get mask for predictions for the current class
                                class_mask = (max_logit_idx_curr == i)

                                # set to all False
                                class_confidence_mask_i = torch.zeros(max_logit_val_curr.size(), dtype=torch.bool).to(dev)

                                # if mean is not higher than self.consis_confidence_thresh, use that as default thresh
                                thresh = max(self.per_class_confidence_thresh[i].get_mean(), self.consis_confidence_thresh)

                                # print("Class,", i, ",thresh, ", thresh)
                                # put true in appropriate places

                                class_confidence_mask_i[class_mask] = (max_logit_val_curr[class_mask] > thresh)

                                if confidence_mask is not None:
                                    confidence_mask = confidence_mask | class_confidence_mask_i
                                else:
                                    confidence_mask = class_confidence_mask_i
                        else:
                            confidence_mask = (max_logit_val_curr > self.consis_confidence_thresh) 
                        
                        # print("Num pixles in PL confidence", confidence_mask.sum())

                        #final mask
                        consis_confidence_mask = (consis_mask & confidence_mask)

                        # print("Num pixels in PL AFTER BOTH MASK:", consis_confidence_mask.sum())

                        if i == 0 and DEBUG:
                            consistent = pltki[0].clone().detach()
                            consistent[~consis_mask] = 255

                            confidence = pltki[0].clone().detach()
                            confidence[~confidence_mask] = 255

                            subplotimg(axs[6][0], consistent, 'Consis Filter', cmap="cityscapes")
                            subplotimg(axs[6][1], confidence, 'Confidence Filter', cmap="cityscapes")
                        # 255 or zero out all pixels not in mask
                    
                        pltki[0][~consis_confidence_mask]= 255
                        pseudo_weight_warped_i[~consis_confidence_mask] = 0

                        
                        if i == 0 and DEBUG:
                            subplotimg(axs[6][2], pltki[0], 'Consis Confidence Filter', cmap="cityscapes")



                        # populate Circular Tensors

                        if self.consis_confidence_per_class_thresh:

                            # update buffers per class
                            for i in range(self.num_classes):
                                
                                class_mask = (pltki[0] == i)

                                num_preds_class_i = class_mask.sum().item()

                                # we will choose 50 pixels from each class to get logit values from
                                random_sample_num = min(num_preds_class_i, 50)

                                #if no samples in PL
                                if random_sample_num == 0:
                                    continue

                                true_indices = torch.where(class_mask)

                                change_indices = np.random.choice(len(true_indices[0]), size=random_sample_num, replace=False)

                                samples = max_logit_val_curr[true_indices[0][change_indices], true_indices[1][change_indices]]
                                
                                # print("Before", self.per_class_confidence_thresh[i].get_mean())
                                self.per_class_confidence_thresh[i].append(samples)
                                # print("After", self.per_class_confidence_thresh[i].get_mean())

                    pseudo_weight_warped.append(pseudo_weight_warped_i)
                    pseudo_label_warped.append(pltki[0])
                
                pseudo_weight_warped = torch.stack(pseudo_weight_warped)
                pseudo_label_warped = torch.stack(pseudo_label_warped)
                if DEBUG:
                    subplotimg(axs[1][6], pseudo_label_warped[0], 'Original PL Warped', cmap="cityscapes")

                if self.consis_filter_rare_class:
                    pseudo_label_warped = rare_class_or_filter(pseudo_label, pseudo_label_warped, rare_common_compare=True)
                    pseudo_weight_warped[pseudo_label_warped == 255] = 0

                if self.oracle_mask:
                    if DEBUG:
                        subplotimg(axs[2][4], pseudo_label_warped[0], 'PL before warp', cmap="cityscapes")
                    # Let's do standard warp.  No need to fill in.  And mask out the weight whereever it's wrong
                    oracle_map = (pseudo_label_warped == target_img_extra["gt_semantic_seg"].squeeze(1)) & (target_img_extra["gt_semantic_seg"].squeeze(1) != 255)
                    pseudo_weight_warped[~oracle_map] = 0
                    pseudo_label_warped[~oracle_map] = 255
                    if DEBUG:
                        subplotimg(axs[3][4], (oracle_map[[0]]).repeat(3, 1, 1)*255, 'Oracle Map')

                if self.class_mask_warp is not None:
                    # Flipping thing and stuff here so that the variable name is more intuitive
                    if self.class_mask_warp == "thing":
                        to_mask = CityscapesDataset.STUFF_IDX
                    elif self.class_mask_warp == "stuff":
                        to_mask = CityscapesDataset.THINGS_IDX

                    for thing_class in to_mask:
                        pseudo_label_warped[pseudo_label_warped == thing_class] = 255
                        pseudo_weight_warped[pseudo_label_warped == thing_class] = 0

                if self.warp_cutmix:
                    if self.accel:
                        mixed_im_warp, mixed_lbl_warp, mixed_seg_weight_warp, _, mixed_flow = self.get_mixed_im(pseudo_weight_warped, pseudo_label_warped, img, target_img, gt_semantic_seg, means, stds, img_extra["flow[0]"], target_img_extra["flow[0]"])
                        custom_loss = self.model(mixed_im_warp, target_img_metas, mixed_lbl_warp.view(B, 1, H, W), seg_weight=mixed_seg_weight_warp, flow=mixed_flow)
                    else:
                        mixed_im_warp, mixed_lbl_warp, mixed_seg_weight_warp, _ = self.get_mixed_im(pseudo_weight_warped, pseudo_label_warped, img, target_img, gt_semantic_seg, means, stds)
                        custom_loss = self.model(mixed_im_warp, target_img_metas, mixed_lbl_warp.view(B, 1, H, W), seg_weight=mixed_seg_weight_warp)

                    if DEBUG:
                        subplotimg(axs[1, 4], invNorm(mixed_im_warp[0]), "Warped Im with CutMix")
                        # subplotimg(axs[1, 5], mixed_im[0])
                        subplotimg(axs[1, 5], mixed_lbl_warp[0], "Warped Label with CutMix", cmap="cityscapes")
                        subplotimg(axs[1, 6], mixed_seg_weight_warp[0].repeat(3, 1, 1)*255)

                elif self.exclusive_warp_cutmix:
                    # choice = np.random.choice([0, 1], p=[0.5, 0.5])
                    choice = 0 if random.uniform(0, 1) > 0.5 else 1
                    if choice == 0: # Warp but no cutmix
                        custom_loss = self.model(
                            target_img,
                            target_img_metas,
                            pseudo_label_warped.view(B, 1, H, W),
                            seg_weight=pseudo_weight_warped
                        )
                    elif choice == 1: # Cutmix but no warp
                        if self.accel:
                            mixed_img, mixed_lbl, mixed_seg_weight, mix_masks, mixed_flow = self.get_mixed_im(pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds, img_extra["flow[0]"], target_img_extra["flow[0]"])
                            custom_loss = self.model(mixed_img, img_metas, mixed_lbl, seg_weight=mixed_seg_weight, return_feat=False, flow=mixed_flow)
                        else:
                            mixed_img, mixed_lbl, mixed_seg_weight, mix_masks = self.get_mixed_im(pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds)
                            custom_loss = self.model(mixed_img, img_metas, mixed_lbl, seg_weight=mixed_seg_weight, return_feat=False)

                        if DEBUG:
                            subplotimg(axs[0, 4], invNorm(mixed_img[0]), "Mixed Im with CutMix")
                            subplotimg(axs[0, 5], mixed_lbl[0], "Mixed lbl with CutMix", cmap="cityscapes")
                            subplotimg(axs[0, 6], mixed_seg_weight[0].repeat(3, 1, 1)*255)
                        # Train on mixed images
                        # seg_debug['Mix'] = self.get_model().debug_output
                        # mix_losses = add_prefix(mix_losses, 'mix')
                        # mix_loss, mix_log_vars = self._parse_losses(mix_losses)
                        # log_vars.update(mix_log_vars)
                        # (mix_loss * self.l_mix_lambda).backward()
                else:
                    if self.accel:
                        custom_loss = self.model(target_img, target_img_metas, pseudo_label_warped.view(B, 1, H, W), seg_weight=pseudo_weight_warped)
                    else:
                        custom_loss = self.model(target_img, target_img_metas, pseudo_label_warped.view(B, 1, H, W), seg_weight=pseudo_weight_warped, flow=target_img_extra["flow[0]"])
                warped_pl_loss, warped_pl_log_vars = self._parse_losses(custom_loss)
                warped_pl_loss = warped_pl_loss * self.l_warp_lambda
                log_vars["L_warp"] = warped_pl_log_vars["loss"]

                warped_pl_loss.backward() #NOTE: do we need to retain graph?

                if self.local_iter % 100 == 0:
                    self.init_cml_debug_metrics()
                self.add_training_debug_metrics(pseudo_label, pseudo_label_warped, pseudo_weight_warped, target_img_extra, log_vars)

            if self.oracle_mask_add_noise or self.oracle_mask_remove_pix:
                # oracle masking
                oracle_map = (pseudo_label == target_img_extra["gt_semantic_seg"].squeeze(1)) & (target_img_extra["gt_semantic_seg"].squeeze(1) != 255)
                pseudo_label_oracle_consis = torch.clone(pseudo_label)
                pseudo_label_oracle_consis_weight = torch.clone(pseudo_weight)

                pseudo_label_oracle_consis_weight[~oracle_map] = 0
                pseudo_label_oracle_consis[~oracle_map] = 255

                inconsis_pixels = (pseudo_label != target_img_extra["gt_semantic_seg"].squeeze(1)) & (target_img_extra["gt_semantic_seg"].squeeze(1) != 255)

                for i in range(batch_size):
                    

                    if self.oracle_mask_add_noise:
                        inconsis_pix_i = inconsis_pixels[i]
                        # print("Before non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())
                        # print(self.oracle_mask_noise_percent)

                        num_inconsis = inconsis_pix_i.sum().item()
                        num_noise = min(int(self.oracle_mask_noise_percent * oracle_map[i].sum().item()), num_inconsis)

                        num_to_change = num_inconsis - num_noise

                        true_indices = torch.where(inconsis_pix_i)

                        change_indices = np.random.choice(len(true_indices[0]), size=num_to_change, replace=False)
                    
                        inconsis_pix_i[true_indices[0][change_indices], true_indices[1][change_indices]] = False

                        # print("after", inconsis_pix_i.sum().item())

                        pseudo_label_oracle_consis[i][inconsis_pix_i] = pseudo_label[i][inconsis_pix_i]
                        pseudo_label_oracle_consis_weight[i][inconsis_pix_i] = pseudo_weight[i][inconsis_pix_i]

                        # print("after non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())

                    if self.oracle_mask_remove_pix:
                        # print("Before non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())
                        num_to_change = int(self.oracle_mask_noise_percent * oracle_map[i].sum().item())

                        true_indices = torch.where(oracle_map[i])

                        change_indices = np.random.choice(len(true_indices[0]), size=num_to_change, replace=False)

                        oracle_map[i][true_indices[0][change_indices], true_indices[1][change_indices]] = False

                        pseudo_label_oracle_consis[i][~oracle_map[i]] = 255
                        pseudo_label_oracle_consis_weight[i][~oracle_map[i]] = 0

                        # print("after non zero", (pseudo_label_oracle_consis[i] != 255).sum().item())

                pseudo_label = pseudo_label_oracle_consis
                pseudo_weight = pseudo_label_oracle_consis_weight


            if self.l_mix_lambda > 0:
                # Apply mixing
                if self.accel:
                    mixed_img, mixed_lbl, mixed_seg_weight, mix_masks, mixed_flow = self.get_mixed_im(pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds, img_extra["flow[0]"], target_img_extra["flow[0]"])
                    mix_losses = self.model(mixed_img, img_metas, mixed_lbl, seg_weight=mixed_seg_weight, return_feat=False, return_logits=True, flow=mixed_flow)
                else:
                    mixed_img, mixed_lbl, mixed_seg_weight, mix_masks = self.get_mixed_im(pseudo_weight, pseudo_label, img, target_img, gt_semantic_seg, means, stds, None, None)
                    mix_losses = self.model(mixed_img, img_metas, mixed_lbl, seg_weight=mixed_seg_weight, return_feat=False, return_logits=True)

                if DEBUG:
                    subplotimg(axs[0, 4], invNorm(mixed_img[0]), "Mixed Im with CutMix")
                    subplotimg(axs[0, 5], mixed_lbl[0], "Mixed lbl with CutMix", cmap="cityscapes")
                    subplotimg(axs[0, 6], mixed_seg_weight[0].repeat(3, 1, 1)*255)
                # Train on mixed images
                # breakpoint()
                if isinstance(mix_losses['decode.logits'], tuple):
                    mix_student_pred = torch.softmax(mix_losses['decode.logits'][0].detach(), dim=1) #indexing 0th to get the fused predictions - which is what is used in decode_head.forward_test
                else:
                    mix_student_pred = torch.softmax(mix_losses['decode.logits'].detach(), dim=1) #For non HRDA don't index into tuple
                mixed_gt = torch.cat(self.get_mixed_gt(mix_masks, gt_semantic_seg, target_img_extra))
                mix_student_pred = resize(
                    input=mix_student_pred,
                    size=mixed_gt.shape[2:],
                    mode='bilinear',
                    align_corners=self.get_model().align_corners)
                _, mix_student_pred = torch.max(mix_student_pred, dim=1)
                self.add_iou_metrics(mix_student_pred, mixed_gt, log_vars, name="Mixed Student")
                del mix_losses['decode.logits'], mix_student_pred, mixed_gt
                seg_debug['Mix'] = self.get_model().debug_output
                mix_losses = add_prefix(mix_losses, 'mix')
                mix_loss, mix_log_vars = self._parse_losses(mix_losses)
                log_vars.update(mix_log_vars)
                (mix_loss * self.l_mix_lambda).backward()
            
            # Masked Training
            if self.enable_masking and self.mask_mode.startswith('separate'):
                # assert not self.accel, "Score fusion not supported with MIC masking"
                masked_loss = self.mic(self.model, img, img_metas,
                                    gt_semantic_seg, target_img,
                                    target_img_metas, valid_pseudo_mask,
                                    pseudo_label, pseudo_weight, img_extra["flow[0]"], target_img_extra["flow[0]"])
                seg_debug.update(self.mic.debug_output)
                masked_loss = add_prefix(masked_loss, 'masked')
                masked_loss, masked_log_vars = self._parse_losses(masked_loss)
                log_vars.update(masked_log_vars)
                masked_loss.backward()


            if DEBUG:
                subplotimg(axs[0][0], invNorm(target_img_extra["img"][0]), 'Current Img batch 0')
                subplotimg(axs[0][1], invNorm(target_img_extra["imtk"][0]), 'Future Imtk batch 0')
                subplotimg(axs[1][0], pseudo_label_warped[0], 'PL Warped', cmap="cityscapes")
                # subplotimg(axs[1][1], pseudo_label_warped[1], 'PL Warped2', cmap="cityscapes")
                subplotimg(axs[1][1], pseudo_label[0], 'PL Plain', cmap="cityscapes")
                subplotimg(axs[1][2], pseudo_label_fut[0], 'PL Plain FUT', cmap="cityscapes")
                # subplotimg(axs[1][2], pseudo_label[1], 'PL Plain2', cmap="cityscapes")
                subplotimg(axs[3][1], pseudo_weight_warped[[0]].repeat(3, 1, 1) * 255, 'Warped PL Weight')

                target_img_gt_semantic_seg = target_img_extra["gt_semantic_seg"][0, 0]

                tolog = f"L_warp: {warped_pl_loss.item():.2f}\n"
                if self.l_mix_lambda > 0:
                    tolog += f"L_mix: {mix_loss.item():.2f}\n"

                warp_iou, warp_miou, warp_iou_mask = self.fast_iou(pseudo_label_warped[0].cpu().numpy(), target_img_gt_semantic_seg.cpu().numpy(), custom_mask=pseudo_weight_warped[0].cpu().bool().numpy())
                tolog += f"Warp PL miou: {warp_miou:.2f}\n"
                plain_iou, plain_miou, plain_iou_mask = self.fast_iou(pseudo_label[0].cpu().numpy(), target_img_gt_semantic_seg.cpu().numpy(), custom_mask=pseudo_weight[0].cpu().bool().numpy())
                tolog += f"Plain PL miou: {plain_miou:.2f}\n"
                pl_agreement_iou, pl_agreement_miou, _ = self.fast_iou(pseudo_label[0].cpu().numpy(), pseudo_label_warped[0].cpu().numpy(), custom_mask=masks[0])
                tolog += f"PL Agree miou: {pl_agreement_miou:.2f}\n"
                # ax[3][0].bar(plain_iou, CityscapesDataset.CLASSES)
                # ax[3][0].bar(warp_iou, CityscapesDataset.CLASSES)
                
                multiBarChart({"warp_iou": warp_iou, "plain_iou": plain_iou, "agreement_iou": pl_agreement_iou}, CityscapesDataset.CLASSES, title="Per class IoU", xlabel="Class", ylabel="IoU", ax=large_axs[0][0])

                plot_confusion_matrix(confusion_matrix(pseudo_label_warped[0].cpu().numpy(), target_img_gt_semantic_seg.cpu().numpy(), num_classes=len(CityscapesDataset.CLASSES)), large_axs[1][0], class_names=CityscapesDataset.CLASSES, title="Warp PL Confusion Matrix", normalize=True)


                axs[2][0].text(
                    0.1,
                    0.5,
                    tolog
                )

                subplotimg(
                    axs[1][3],
                    target_img_gt_semantic_seg.cpu().numpy(), 'GT',
                    cmap="cityscapes"
                )

                subplotimg(
                    axs[2][1],
                    get_segmentation_error_vis(pseudo_label_warped[0].cpu().numpy(), target_img_gt_semantic_seg.cpu().numpy()), 'PL Warped Error',
                    cmap="cityscapes"
                )

                subplotimg(
                    axs[2][2],
                    get_segmentation_error_vis(pseudo_label[0].cpu().numpy(), target_img_gt_semantic_seg.cpu().numpy()), 'Plain PL Error',
                    cmap="cityscapes"
                )

                #plot the pl agreement error
                subplotimg(
                    axs[2][3],
                    get_segmentation_error_vis(pseudo_label[0].cpu().numpy(), pseudo_label_warped[0].cpu().numpy()), 'PL Agreement Error',
                    cmap="cityscapes"
                )
                
                subplotimg(axs[4, 0],
                    visFlow(target_img_extra["flow"][0].permute(1, 2, 0).cpu().numpy(), image=invNorm(target_img_extra["img"][0]).permute(1, 2, 0).cpu().numpy(), skip_amount=200)
                )

                # subplotimg(
                #     axs[1][3],
                #     plain_iou_mask.repeat(3, 1, 1) * 255, "plain iou mask"
                # )

                # subplotimg(
                #     axs[1][4],
                #     warp_iou_mask.repeat(3, 1, 1) * 255, "warp iou mask"
                # )

                fig.savefig(f"work_dirs/LWarpPLAnalysis/ims{self.local_iter}.png", dpi=200)
                # fig.close()
                large_fig.savefig(f"work_dirs/LWarpPLAnalysis/graphs{self.local_iter}.png", dpi=200)
                # large_fig.close()
                # breakpoint()

        self.local_iter += 1

        return log_vars
