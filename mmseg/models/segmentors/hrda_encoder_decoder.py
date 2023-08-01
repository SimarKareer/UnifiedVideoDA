# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Add upscale_pred flag
# - Update debug_output system
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
import torch
import copy

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from tools.aggregate_flows.flow.my_utils import backpropFlow


def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img


@SEGMENTORS.register_module()
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scales=[1],
                 hr_crop_size=None,
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1,
                 multimodal=False):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg, 
            multimodal=multimodal)

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop

    def extract_unscaled_feat(self, img, masking_branch = None):
        if self.multimodal:
            x = self.backbone(img, masking_branch)
        else:
            x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def generate_pseudo_label(self, img, img_metas, flow=None):
        self.update_debug_state()
        out = self.encode_decode(img, img_metas, flow=flow)
        if self.debug:
            self.debug_output = self.decode_head.debug_output
        return out
    
    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference.
        x: x[low/highres, mmbranch, segformer_layer] #This might change in different calls of _decode_head_forward_test
                          [B, C, H, W]
            - x[0][0][0]: [2, 64, 128, 128]
            - x[0][0][1]: [2, 128, 64, 64]
            - x[0][0][2]: [2, 320, 32, 32]
            - x[0][0][3]: [2, 512, 16, 16]
        """
        if self.multimodal:
            # scale0branch0 = x[0][0]
            # scale0branch1 = x[0][1]
            # scale1branch0 = {"features": x[1]['features'][0], "boxes": x[1]['boxes']}
            # scale1branch1 = {"features": x[1]['features'][1], "boxes": x[1]['boxes']}
            x = [ 
                [x[0][0], {"features": x[1]['features'][0], "boxes": copy.copy(x[1]['boxes'])}], #Need to copy boxes, as it is modified in-place by self.decode_head.forward_test
                [x[0][1], {"features": x[1]['features'][1], "boxes": copy.copy(x[1]['boxes'])}]  #Need to copy boxes, as it is modified in-place by self.decode_head.forward_test
                ] #Now x[i] corresponds to branch i
            seg_logits = [self.decode_head.forward_test(x[i], img_metas, self.test_cfg) for i in range(self.backbone.num_parallel)]
            seg_logits = self._get_ensemble_logits(seg_logits)
        else:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_logits=False, reset_crop=True):
        """Run forward function and calculate loss for decode head in
        training.
        x: x[low/highres, mmbranch, segformer_layer] #This might change in different calls of _decode_head_forward_test
                      [B, C,  H,   W]
        - x[0][0][0]: [2, 64, 128, 128]
        - x[0][0][1]: [2, 128, 64, 64]
        - x[0][0][2]: [2, 320, 32, 32]
        - x[0][0][3]: [2, 512, 16, 16]
        
        return: losses: dict{}
        """
        losses = dict()
        if self.multimodal:
            loss_decode_list = []
            for i in range(len(x)):
                loss_decode_list.append(self.decode_head.forward_train([x[0][i], x[1][i]], img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        seg_weight=seg_weight, return_logits=True, reset_crop=False))
            # loss_decode_list[loss_dict_number low or high res]["dict_key"]
            ens_logit = [] # list over each of the 3 types of logits, (I think hr, lr, fuse but don't know the order)
            for res_num in range(3):
                ens_logit.append(self._get_ensemble_logits([loss_decode_list[i]['logits'][res_num] for i in range(self.backbone.num_parallel)]))

            ens_logit = tuple(ens_logit)
            ens_loss = self.decode_head.losses(ens_logit, gt_semantic_seg, seg_weight)
            loss_decode_list.append(ens_loss)
            loss_decode = self._average_losses(loss_decode_list)
            if return_logits:
                loss_decode['logits'] = ens_logit
            
            self.decode_head.reset_crop()
        else:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        seg_weight, return_logits, reset_crop=reset_crop)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def encode_decode(self, img, img_metas, upscale_pred=True, flow=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = self.resize(img, s)
            if i >= 1 and self.hr_slide_inference:
                mres_feats.append(self.extract_slide_feat(scaled_img))
            else:
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
        out = self._decode_head_forward_test(mres_feats, img_metas)
        
        #NOTE: basically perform score fusion after the sliding eval
        if flow is not None:
            assert self.accel, "Only ACCELHRDAEncoderDecoder supports flow"
            #NOTE: forget to align preds via flow before score fusion
            out2 = out[out.shape[0]//2:]
            out2 = resize(out2, size=flow.shape[2:], mode='bilinear')
            out2 = backpropFlow(flow.permute((0, 2, 3, 1)), out2.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            out2 = resize(out2, size=out.shape[2:], mode='bilinear')

            out = torch.cat([out[:out.shape[0]//2], out[out.shape[0]//2:]], dim=1)
            out = self.sf_layer(out)

        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _forward_train_features(self, img, masking_branch=None):
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img, masking_branch))
        return mres_feats, prob_vis

    def forward(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False,
                      masking_branch=None,
                      reset_crop=True):
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
            masking_branch: what branches to mask (modality dropout)

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()

        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img, masking_branch)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        loss_decode = self._decode_head_forward_train(mres_feats, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      return_logits, reset_crop=reset_crop)
        losses.update(loss_decode)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
        self.local_iter += 1
        return losses

    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}
