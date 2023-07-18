import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from .cityscapes import CityscapesDataset
from .seqUtils import SeqUtils
import cityscapesscripts.helpers.labels as CSLabels
import pdb

@DATASETS.register_module()
class CityscapesSeqDataset(SeqUtils, CityscapesDataset):
    """Cityscapes-Seq dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelIds.png' for Cityscapes-seq dataset.
    """

    def __init__(self, split, load_gt, img_suffix='_leftImg8bit.png', seg_map_suffix='_gtFine_labelTrainIds.png', frame_offset=1, flow_dir=None, crop_pseudo_margins=None, data_type="rgb", **kwargs):
        # breakpoint()

        CityscapesDataset.__init__(
            self, #must explicitly pass self
            split,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            crop_pseudo_margins=None,
            load_annotations=False,
            **kwargs)
        SeqUtils.__init__(self)
        
        # breakpoint()
        self.unpack_list = "train" in split
        # assert(crop_pseudo_margins is not None)
        if crop_pseudo_margins:
            assert kwargs['pipeline']["im_pipeline"][-1]['type'] == 'Collect'
            kwargs['pipeline']["im_pipeline"][-1]['keys'].append('valid_pseudo_mask')
            # if crop_pseudo_margins is not None:
            self.pseudo_margins = crop_pseudo_margins
        else:
            self.pseudo_margins = None

        self.seq_imgs = []
        self.seq_flows = []
        self.frame_offset = frame_offset
        self.load_gt = load_gt
        self.flow_dir = flow_dir
        for i, offset in enumerate(frame_offset):
            self.seq_imgs.append(self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=offset))
            seq_flow = None if flow_dir == None or flow_dir[i] == None else self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, img_prefix=flow_dir[i], frame_offset=offset)
            self.seq_flows.append(seq_flow)
        
        zero_index = frame_offset.index(0)
        assert zero_index != -1, "Need zero index"
        self.img_infos = self.seq_imgs[zero_index]
        self.zero_index = zero_index

        self.ds = "cityscapes-seq"
        
        self.data_type = data_type

        # self.fut_images, self.img_infos, self.flows = self.cofilter_img_infos(self.fut_images, self.img_infos, self.flows, self.img_dir, flow_dir)

        self.palette_to_id = [(k, i) for i, k in enumerate(self.PALETTE)]

        
        viperClasses = ("unlabeled", "ambiguous", "sky","road","sidewalk","railtrack","terrain","tree","vegetation","building","infrastructure","fence","billboard","traffic light","traffic sign","mobilebarrier","firehydrant","chair","trash","trashcan","person","animal","bicycle","motorcycle","car","van","bus","truck","trailer","train","plane","boat")

        self.label_map = {k: 200 for k in range(255)}
        self.adaptation_map = {k: 201 for k in range(255)}
        
        for _, label in CSLabels.trainId2label.items():
            name = label.name
            if name in viperClasses:
                self.adaptation_map[viperClasses.index(name)] = label.trainId

        for _, label in CSLabels.trainId2label.items():
            self.label_map[label.id] = label.trainId
        
        self.label_map = None
        
        # After label map: license plate, road, sidewalk, ...    unlabelled
        #                  -1             0,     1,              255
    # def __getitem__(self, idx):
    #     print("CityscapesSeq __getitem__")
    #     super().__getitem__(idx)
