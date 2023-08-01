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
import torch.nn as nn
import copy

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS
from .hrda_encoder_decoder import HRDAEncoderDecoder

from tools.aggregate_flows.flow.my_utils import backpropFlow


@SEGMENTORS.register_module()
class ACCELHRDAEncoderDecoder(HRDAEncoderDecoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf_layer = self.get_score_fusion_layer(self.num_classes)
        self.accel = True

    def get_score_fusion_layer(self, num_classes):
        sf_layer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.zeros_(sf_layer.weight)
        nn.init.eye_(sf_layer.weight[:, :num_classes, :, :].squeeze(-1).squeeze(-1))
        return sf_layer

    def forward_with_aux(self, img, img_metas, flow):
        return self.forward(img, img_metas, None, flow, with_aux=True, return_logits=True)

    def forward(self,
                img,
                img_metas,
                gt_semantic_seg,
                flow,
                seg_weight=None,
                return_feat=False,
                return_logits=False,
                masking_branch=None,
                with_aux=False):
        """
            at this point inputs will have double batch size.  This function should break that apart, run one with grad, one without.  Then fuse the results together.
        """

        img_with_grad = img[:img.shape[0]//2] #main img
        img_without_grad = img[img.shape[0]//2:] #second aux img
    
        with torch.no_grad():
            if with_aux:
                without_grad = super().forward_with_aux(img_without_grad, img_metas)
            else:
                without_grad = super().forward(img_without_grad, img_metas, gt_semantic_seg, seg_weight=seg_weight, return_feat=return_feat, return_logits=True, masking_branch=masking_branch)

        if with_aux:
            with_grad = super().forward_with_aux(img_with_grad, img_metas)
        else:
            with_grad = super().forward(img_with_grad, img_metas, gt_semantic_seg, seg_weight=seg_weight, return_feat=return_feat, return_logits=True, masking_branch=masking_branch, reset_crop=False)

        # breakpoint()
        fused_logits = []

        logit_name = "main" if with_aux else "decode.logits"
        for i in range(3):
            # NOTE: I'm upscaling the image here, but normally it's done in decode_head.losses
            aux_im_logits = without_grad[logit_name][i]
            aux_im_logits = resize(aux_im_logits, size=img.shape[2:], mode='bilinear')
            main_im_logits = resize(with_grad[logit_name][i], size=img.shape[2:], mode='bilinear')
            propped_logits = backpropFlow(flow.permute((0, 2, 3, 1)), aux_im_logits.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            cat_logits = torch.cat([main_im_logits, propped_logits], dim=1)
            
            fused_logits.append(self.sf_layer(cat_logits))
        fused_logits = tuple(fused_logits) #need tuple bc dacs has some hard coded tuple things



        # NOTE: I'm returning the main images features, instead of fusing them.  I don't think flow prop at a feature level is a good idea
        merged = {}
        if return_feat:
            merged["features"] = with_grad["features"]
        if return_logits:
            merged[logit_name] = fused_logits
        if not with_aux:
            loss = self.decode_head.losses(fused_logits, gt_semantic_seg, seg_weight)
            merged.update(add_prefix(loss, 'decode'))
            self.decode_head.reset_crop()
        

        return merged
    