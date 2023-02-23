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

    def __init__(self, split, img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', frame_offset=1, flow_dir=None, **kwargs):
        SynthiaDataset.__init__(
            self, #must explicitly pass self
            split=split,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            load_annotations=False,
            **kwargs)
        SeqUtils.__init__(self)
        
        self.flow_dir = flow_dir
        self.fut_images = self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=-1)
        self.img_infos = self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=0)
        self.flows = None if self.flow_dir == None else self.load_annotations_seq(self.img_dir, ".png", self.ann_dir, self.seg_map_suffix, self.split, frame_offset=0)


        self.unpack_list = "train" in split


        self.palette_to_id = [(k, i) for i, k in enumerate(self.PALETTE)]

        # synthiaSeqClasses = #fill in

        CSClasses = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

        synthiaSeq_to_cs = {k: 201 for k in range(255)}
        for i, k in enumerate(CSClasses):
            if k in synthiaSeqClasses:
                cs_to_sythiaSeq[i] = synthiaSeqClasses.index(k)
        
        synthiaSeq_to_cs = {k: 201 for k in range(255)}
        for i, k in enumerate(synthiaSeqClasses):
            if k in CSClasses:
                synthiaSeq_to_cs[i] = CSClasses.index(k)
        self.convert_map = {"cityscapes_synthiaSeq": synthiaSeq_to_cs, "synthiaSeq_cityscapes": synthiaSeq_to_cs}
        self.label_space = "synthiaSeq"
