from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.advseg import AdvSeg
from mmseg.models import UDA
from torch import nn, optim
from torch.autograd import Variable


from mmseg.core import add_prefix
from mmseg.models.uda.fcdiscriminator import FCDiscriminator
import torch
from mmseg.models import UDA, HRDAEncoderDecoder
from mmseg.ops import resize
import torch.nn.functional as F


@UDA.register_module()
class DACSAdvseg(DACS):

    def __init__(self, **cfg):
        super().__init__(**cfg)

        self.local_iter = 0
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.max_iters = cfg['max_iters']
        self.lr_D = cfg['lr_D']
        self.lr_D_power = cfg['lr_D_power']
        self.lr_D_min = cfg['lr_D_min']
        self.discriminator_type = cfg['discriminator_type']
        self.lambda_adv_target = cfg['lambda_adv_target']
        self.adv_scale = cfg['adv_scale']
        self.mask_mode = cfg['mask_mode']
        self.video_discrim = cfg['video_discrim']

        self.model_D = nn.ModuleDict()
        self.optimizer_D = {}
        for k in ['main', 'aux'] if self.model.with_auxiliary_head \
                else ['main']:
            
            self.model_D[k] = FCDiscriminator(num_classes=self.num_classes * 2) if self.video_discrim else FCDiscriminator(num_classes=self.num_classes)
            self.model_D[k].train()
            self.model_D[k].cuda()

            self.optimizer_D[k] = optim.Adam(
                self.model_D[k].parameters(), lr=self.lr_D, betas=(0.9, 0.99))
            self.optimizer_D[k].zero_grad()

        if self.discriminator_type == 'Vanilla':
            self.loss_fn_D = torch.nn.BCEWithLogitsLoss()
        elif self.discriminator_type == 'LS':
            self.loss_fn_D = torch.nn.MSELoss()
        else:
            raise NotImplementedError(self.discriminator_type)

        
    
    def adjust_learning_rate_D(self, optimizer, i_iter):
        coeff = (1 - i_iter / self.max_iters)**self.lr_D_power
        lr = (self.lr_D - self.lr_D_min) * coeff + self.lr_D_min
        assert len(optimizer.param_groups) == 1
        optimizer.param_groups[0]['lr'] = lr
    
    def train_step(self, data_batch, optimizer, **kwargs):

        optimizer.zero_grad()
        for k in self.optimizer_D.keys():
            self.optimizer_D[k].zero_grad()
            self.adjust_learning_rate_D(self.optimizer_D[k], self.local_iter)
        assert len(kwargs) == 0, "kwargs not empty"

        log_vars1 = super().forward_train(**data_batch)
        log_vars2 = self.forward_train_videodisc(**data_batch)
        optimizer.step()
        for k in self.optimizer_D.keys():
            self.optimizer_D[k].step()
        

        # log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        log_vars1.update(log_vars2)
        log_vars1["Local Iteration"] = self.local_iter
        outputs = dict(
            log_vars=log_vars1, num_samples=len(data_batch['img_metas']))
        return outputs


        #Return log1 and log2 merged where log1 and log2 are dictionaries
        # return {**log1, **log2}



    def _resize_preds(self, pred, pred_trg, img):
        if isinstance(self.model.module, HRDAEncoderDecoder):
            for k in pred.keys():
                pred[k] = pred[k][0]
                assert self.model.module.feature_scale == 0.5
                pred[k] = resize(
                    input=pred[k],
                    size=[
                        int(e * self.model.module.feature_scale)
                        for e in img.shape[2:]
                    ],
                    mode='bilinear',
                    align_corners=self.model.module.align_corners)
            for k in pred_trg.keys():
                pred_trg[k] = pred_trg[k][0]
                pred_trg[k] = resize(
                    input=pred_trg[k],
                    size=[
                        int(e * self.model.module.feature_scale)
                        for e in img.shape[2:]
                    ],
                    mode='bilinear',
                    align_corners=self.model.module.align_corners)
    
    def forward_train_videodisc(self, img, img_metas, img_extra, target_img, target_img_metas, target_img_extra):
        source_label = 0
        target_label = 1

        # self.update_debug_state()
        # seg_debug = {}
        log_vars = dict()

        for param in self.model_D.module.parameters():
            param.requires_grad = False

        pred = self.model.module.forward_with_aux(img, img_metas)
        pred_tk = self.model.module.forward_with_aux(img_extra["imtk"], img_extra["imtk_metas"])
        pred_trg = self.model.module.forward_with_aux(target_img, target_img_metas)
        pred_trg_tk = self.model.module.forward_with_aux(target_img_extra["imtk"], target_img_extra["imtk_metas"])

        if isinstance(self.model.module, HRDAEncoderDecoder):
            self.model.module.decode_head.reset_crop()
        
        self._resize_preds(pred, pred_trg, img)
        self._resize_preds(pred_tk, pred_trg_tk, img_extra["imtk"])



        g_trg_losses = dict()
        for k in pred_trg.keys():
            if self.video_discrim:
                concat_feats = torch.cat((pred_trg[k], pred_trg_tk[k]), dim=1)
                D_out = self.model_D.module[k](F.softmax(concat_feats, dim=1))
            else:
                D_out = self.model_D.module[k](F.softmax(pred_trg[k], dim=1))
                
            loss_G = self.loss_fn_D(
                D_out,
                Variable(
                    torch.FloatTensor(
                        D_out.data.size()).fill_(source_label)).cuda())
            # remember to have the word 'loss' in key
            g_trg_losses[
                f'G_trg.loss.{k}'] = self.lambda_adv_target[k] * loss_G
        g_trg_loss, g_trg_log_vars = self._parse_losses(g_trg_losses)
        g_trg_loss = self.adv_scale * g_trg_loss
        g_trg_loss.backward()

        #######################################################################
        # Train Discriminator
        #######################################################################
        # bring back requires_grad
        for param in self.model_D.module.parameters():
            param.requires_grad = True

        # train with source
        d_src_losses = dict()
        for k in pred.keys():
            if self.video_discrim:
                pred[k] = pred[k].detach()
                pred_tk[k] = pred_tk[k].detach()
                concat_feats = torch.cat((pred[k], pred_tk[k]), dim=1)
                D_out_src = self.model_D.module[k](F.softmax(concat_feats, dim=1))
            else:
                pred[k] = pred[k].detach()
                D_out_src = self.model_D.module[k](F.softmax(pred[k], dim=1))
            loss_D = self.loss_fn_D(
                D_out_src,
                Variable(
                    torch.FloatTensor(
                        D_out_src.data.size()).fill_(source_label)).cuda())
            d_src_losses[f'D_src.loss.{k}'] = loss_D / 2
        d_src_loss, d_src_log_vars = self._parse_losses(d_src_losses)
        d_src_loss = self.adv_scale * d_src_loss
        d_src_loss.backward()

        # train with target
        d_trg_losses = dict()
        for k in pred_trg.keys():
            if self.video_discrim:
                pred_trg[k] = pred_trg[k].detach()
                pred_trg_tk[k] = pred_trg_tk[k].detach()
                concat_feats = torch.cat((pred_trg[k], pred_trg_tk[k]), dim=1)
                D_out_trg = self.model_D.module[k](F.softmax(concat_feats, dim=1))
            else:
                pred_trg[k] = pred_trg[k].detach()
                D_out_trg = self.model_D.module[k](F.softmax(pred_trg[k], dim=1))
            loss_D = self.loss_fn_D(
                D_out_trg,
                Variable(
                    torch.FloatTensor(
                        D_out_trg.data.size()).fill_(target_label)).cuda())
            d_trg_losses[f'D_trg.loss.{k}'] = loss_D / 2
        d_trg_loss, d_trg_log_vars = self._parse_losses(d_trg_losses)
        d_trg_loss = self.adv_scale * d_trg_loss
        d_trg_loss.backward()

        log_vars["g_trg_loss"] = g_trg_loss.item()
        log_vars["d_src_loss"] = d_src_loss.item()
        log_vars["d_trg_loss"] = d_trg_loss.item()

        return log_vars
