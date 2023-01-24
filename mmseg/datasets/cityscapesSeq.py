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

    def __init__(self, split, img_suffix='_leftImg8bit.png', seg_map_suffix='_gtFine_labelTrainIds.png', frame_offset=1, flow_dir=None, crop_pseudo_margins=None, **kwargs):
        # breakpoint()

        CityscapesDataset.__init__(
            self, #must explicitly pass self
            split,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            crop_pseudo_margins=None,
            # load_annotations=True,
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

        self.flow_dir = flow_dir
        # breakpoint()
        self.past_images = self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=frame_offset)
        self.flows = None if self.flow_dir == None else self.load_annotations_seq(self.img_dir, img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=frame_offset)
        # breakpoint()
        # self.flow_dir = "/srv/share4/datasets/VIPER_Flowv2/train/flow_occ" #TODO Temporary, must fix or will give horrible error
        # self.flow_dir = "/srv/share4/datasets/VIPER_Flow/train/flow"
        self.palette_to_id = [(k, i) for i, k in enumerate(self.PALETTE)]
        # breakpoint()

        
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
