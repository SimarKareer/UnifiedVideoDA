from signal import SIG_DFL
from .builder import DATASETS
from .custom import CustomDataset
from .seqUtils import SeqUtils
from .synthia import SynthiaDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import torch
from mmcv.parallel import DataContainer
import numpy as np
import pdb

@DATASETS.register_module()
class SynthiaSeqDataset(SeqUtils, SynthiaDataset):
    """Synthia Seq dataset with options for loading flow and neightboring frames.
    """

    def __init__(self, split, img_suffix='.png', seg_map_suffix='_labelTrainIds_updated.png', frame_offset=1, flow_dir=None, data_type="rgb", **kwargs):
        SynthiaDataset.__init__(
            self, #must explicitly pass self
            split=split,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            load_annotations=False,
            **kwargs)
        SeqUtils.__init__(self)
        
        self.flow_dir = flow_dir
        self.fut_images = self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=-1)    #forward flow
        self.img_infos = self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=0)
        self.flows = None if self.flow_dir == None else self.load_annotations_seq(self.img_dir, ".png", self.ann_dir, self.seg_map_suffix, self.split, frame_offset=0)

        self.data_type = data_type


        self.unpack_list = "train" in split



