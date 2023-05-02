# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
from torch.nn import Module
import torch.nn as nn


class SSLSimSiamModule(Module):

    def __init__(self, cfg):
        super(SSLSimSiamModule, self).__init__()

        self.out_dim = cfg['ssl_out_dim']
        self.pred_dim = cfg['ssl_pred_dim']
        self.reduced_dim = cfg['ssl_reduced_dim']
        self.prev_dim = 131072 # 512 x 16 x 16

        self.criterion = nn.CosineSimilarity(dim=1)

        self.fc = nn.Sequential(
            nn.Linear(self.prev_dim, self.reduced_dim, bias=False),
            nn.BatchNorm1d(self.reduced_dim),
            nn.ReLU(inplace=True), # first layer
            nn.Linear(self.reduced_dim, self.reduced_dim, bias=False),
            nn.BatchNorm1d(self.reduced_dim),
            nn.ReLU(inplace=True), # second layer
            nn.Linear(self.reduced_dim,self.out_dim),
            nn.BatchNorm1d(self.out_dim, affine=False)
        ) # output layer
        self.fc[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(self.out_dim, self.pred_dim, bias=False),
            nn.BatchNorm1d(self.pred_dim),
            nn.ReLU(inplace=True), # hidden layer
            nn.Linear(self.pred_dim, self.out_dim)
        ) # output layer


    def __call__(self, model, target_img_aug_1, target_imtk_aug_1):
        #concat
        img_tk_concat = torch.cat((target_img_aug_1, target_imtk_aug_1)).cuda()

        # print(img_tk_concat.size())

        img_tk_feats = model.extract_feat(img_tk_concat)

        # for x in img_tk_feats:
        #     print(x.size())

        # print("****")

        img_tk_feats_last_layer = img_tk_feats[-1]
        # print("Last layer size", img_tk_feats_last_layer.size())
        

        img_tk_feats_flattened =  torch.flatten(img_tk_feats_last_layer, start_dim=1)
        # print("flattened size", img_tk_feats_flattened.size())


        half_split = len(img_tk_feats_flattened) //2

        # for i in range(len(img_tk_feats_flattened)):
        #     print(img_tk_feats_flattened[i].size())

        img_feats_flattened = img_tk_feats_flattened[:half_split, :]
        img_tk_feats_flattened = img_tk_feats_flattened[half_split: , :]

        # print("img", img_feats_flattened.size())
        # print("imgtk", img_feats_flattened.size())

        z1 = self.fc(img_feats_flattened)
        z2 = self.fc(img_tk_feats_flattened)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        z1 = z1.detach()
        z2 = z2.detach()

        ssl_loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

        return ssl_loss