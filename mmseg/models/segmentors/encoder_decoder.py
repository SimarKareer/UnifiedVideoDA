# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support for seg_weight and forward_with_aux
# - Update debug output system

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from ..utils.dacs_transforms import get_mean_std
from ..utils.visualization import prepare_debug_out, subplotimg
from .base import BaseSegmentor
import random


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 multimodal=False):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.automatic_debug = True
        self.debug = False
        self.debug_output = {}
        if train_cfg is not None and 'log_config' in train_cfg:
            self.debug_img_interval = train_cfg['log_config']['img_interval']
        self.local_iter = 0
        self.multimodal = multimodal
        if self.multimodal:
            self.alpha = nn.Parameter(torch.ones(self.backbone.num_parallel, requires_grad=True)) #TODO: Make sure alpha is trainable
            self.register_parameter('alpha', self.alpha)
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, masking_branch = None):
        """Extract features from images."""
        if not self.multimodal:
            x = self.backbone(img)
        else:
            x = self.backbone(img, masking_branch)
        if self.with_neck:
            x = self.neck(x)
        return x

    def generate_pseudo_label(self, img, img_metas):
        self.update_debug_state()
        if self.debug:
            self.debug_output = {
                'Image': img,
            }
        out = self.encode_decode(img, img_metas)
        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
            self.debug_output['Pred'] = out.cpu().numpy()

        return out

    def encode_decode(self, img, img_metas, upscale_pred=True, flow=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def forward_with_aux(self, img, img_metas):
        raise NotImplementedError("Currently broken")
        self.update_debug_state()

        ret = {}
        if self.multimodal:
            assert isinstance(x, list) and len(x) == 2, "Multimodal encoder should return a list of \
                                                        two tensors, check self.backbone {}".format(type(self.backbone))
            out = [self._decode_head_forward_test(x[0], img_metas), self._decode_head_forward_test(x[1], img_metas)]
            ens = 0
            alpha_soft = F.softmax(self.alpha)
            for l in range(2):
                ens += alpha_soft[l] * out[l].detach()
            out = ens
        else:    
            x = self.extract_feat(img)
            out = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        ret['main'] = out

        if self.with_auxiliary_head:
            assert not isinstance(self.auxiliary_head, nn.ModuleList)
            if self.multimodal:
                assert isinstance(x, list) and len(x) == 2, "Multimodal encoder should return a list of \
                                                        two tensors, check self.backbone {}".format(type(self.backbone))
                out = [self.auxiliary_head.forward_test(x[0], img_metas), self.auxiliary_head.forward_test(x[0], img_metas)]
                ens = 0
                alpha_soft = F.softmax(self.alpha)
                for l in range(2):
                    ens += alpha_soft[l] * out[l].detach()
                out = ens 
            else:
                out_aux = self.auxiliary_head.forward_test(x, img_metas,
                                                        self.test_cfg)
            out_aux = resize(
                input=out_aux,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ret['aux'] = out_aux

        return ret

    def _average_losses(self, losses):
        """Average losses from different predictions."""
        averaged_loss = {}
        for loss in losses:
            for loss_name, loss_value in loss.items():
                if loss_name not in averaged_loss:
                    if loss_name == "logits":
                        continue
                    averaged_loss[loss_name] = 0
                averaged_loss[loss_name] += loss_value
        for loss_name in averaged_loss:
            averaged_loss[loss_name] /= len(losses)
        return averaged_loss

    def _get_ensemble_logits(self, seg_logits):
        """
        seg_logits: based on this function should be [branch, B, H, W]
        """
        seg_logit = 0
        alpha_soft = F.softmax(self.alpha)
        for l in range(self.backbone.num_parallel):
            seg_logit += alpha_soft[l] * seg_logits[l].detach()
        return seg_logit

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_logits=False):
        """Run forward function and calculate loss for decode head in
        training.
        
        losses: dict{}
        """
        losses = dict()
        if self.multimodal:
            loss_decode_list = []
            for i in range(len(x)):
                loss_decode_list.append(self.decode_head.forward_train(x[i], img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        seg_weight, True))

            ens_logit = self._get_ensemble_logits([loss_decode_list[i]['logits'] for i in range(self.backbone.num_parallel)])
            ens_loss = self.decode_head.losses(ens_logit, gt_semantic_seg, seg_weight)
            loss_decode_list.append(ens_loss)
            loss_decode = self._average_losses(loss_decode_list)
            if return_logits:
                loss_decode['logits'] = ens_logit
        else:
            loss_decode = self.decode_head.forward_train(x, img_metas,
                                                        gt_semantic_seg,
                                                        self.train_cfg,
                                                        seg_weight, return_logits)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        if self.multimodal:
            #x[:, i] = x[low/highres, segformer_layer]
            seg_logits = [self.decode_head.forward_test(x[i], img_metas, self.test_cfg) for i in range(self.backbone.num_parallel)]
            seg_logits = self._get_ensemble_logits(seg_logits)
        else:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        raise NotImplementedError("Currently broken")
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                if self.multimodal:
                    loss_aux_list = []
                    for xi in x:
                        loss_aux_list.append(aux_head.forward_train(xi, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight))
                    loss_aux = self._average_losses(loss_aux_list)
                else:
                    loss_aux = aux_head.forward_train(x, img_metas,
                                                    gt_semantic_seg,
                                                    self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            if self.multimodal:
                loss_aux_list = []
                for xi in x:
                    loss_aux_list.append(self.auxiliary_head.forward_train(xi, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight))
                loss_aux = self._average_losses(loss_aux_list)
            else:
                loss_aux = self.auxiliary_head.forward_train(
                    x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def update_debug_state(self):
        self.debug_output = {}
        if self.automatic_debug:
            self.debug = (self.local_iter % self.debug_img_interval == 0)
        self.decode_head.debug = self.debug
        if self.with_auxiliary_head:
            self.auxiliary_head.debug = self.debug

    def forward(self,
                img,
                img_metas,
                gt_semantic_seg,
                seg_weight=None,
                return_feat=False,
                return_logits=False,
                masking_branch=None):
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

        x = self.extract_feat(img, masking_branch)

        losses = dict()
        if return_feat:
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      return_logits)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        if self.debug:
            self.process_debug(img, img_metas)

        self.local_iter += 1
        return losses

    def forward_train(self,
                    img,
                    img_metas,
                    gt_semantic_seg,
                    seg_weight=None,
                    return_feat=False,
                    return_logits=False,
                    masking_branch=None):
        self.forward(img, img_metas, gt_semantic_seg, seg_weight, return_feat, return_logits, masking_branch)
    
    def process_debug(self, img, img_metas):
        self.debug_output = {
            'Image': img,
            **self.decode_head.debug_output,
        }
        if self.with_auxiliary_head:
            self.debug_output.update(
                add_prefix(self.auxiliary_head.debug_output, 'Aux'))
        if self.automatic_debug:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'encdec_debug')
            os.makedirs(out_dir, exist_ok=True)
            means, stds = get_mean_std(img_metas, img.device)
            for j in range(img.shape[0]):
                rows, cols = 1, len(self.debug_output)
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.92,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                for k, (n, v) in enumerate(self.debug_output.items()):
                    subplotimg(axs[k],
                               **prepare_debug_out(n, v[j], means, stds))
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
            del self.debug_output

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, flow=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        if flow is not None:
            assert self.accel, "Only ACCELHRDAEncoderDecoder supports flow"
            batch_size, _, h_img, w_img = flow.size()
        else:
            batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            if flow is not None:
                flow_imgs = []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    if flow is not None:
                        assert(flow.shape[0] == 1), "Assumed batch size of 1"
                        flow_imgs.append(flow[:, :, y1:y2, x1:x2] if flow is not None else None)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            flow_imgs = torch.cat(flow_imgs, dim=0)
            if flow is not None:
                assert(img.shape[0] == 2 and crop_imgs.shape[0] == 6), "Hardcoded channel reordering for flow"
                # Currently the channels are im1, im1tk, im2, im2tk, im3, im3tk (where each are sliding crops.)  ACCEL needs im1, im2, im3, im1tk, im2tk, im3tk
                crop_imgs = crop_imgs[[0, 2, 4, 1, 2, 3]]

            crop_seg_logits = self.encode_decode(crop_imgs, img_meta, flow=flow_imgs)

            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            if flow is not None:
                assert False, "Didn't implement ACCEL here"
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta, flow=flow)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, flow=flow)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, flow=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, flow=flow)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, flow=flow)
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            output = seg_logit
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, logits=False, flow=None):
        """Simple test with single image."""
        if logits:
            seg_logit = self.inference(img, img_meta, rescale, flow=flow)
            seg_logit = seg_logit.cpu().numpy()
            return seg_logit

        seg_logit = self.inference(img, img_meta, rescale, flow=flow)
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            seg_pred = seg_logit[:, 0]
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
