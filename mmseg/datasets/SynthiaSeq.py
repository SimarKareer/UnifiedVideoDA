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

    def __init__(self, split, load_gt, img_suffix='.png', seg_map_suffix='_labelTrainIds_updated.png', frame_offset=1, flow_dir=None, data_type="rgb", **kwargs):
        SynthiaDataset.__init__(
            self, #must explicitly pass self
            split=split,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            load_annotations=False,
            **kwargs)
        SeqUtils.__init__(self)
        
        self.seq_imgs = []
        self.seq_flows = []
        self.frame_offset = frame_offset
        self.load_gt = load_gt
        self.flow_dir = flow_dir

        # setting indices for all frames used
        self.im_idx = frame_offset[2]
        self.imtk_idx = frame_offset[1]
        self.imtktk_idx = frame_offset[0]
        
        for i, offset in enumerate(frame_offset):
            self.seq_imgs.append(self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=offset))
            seq_flow = None if flow_dir == None or flow_dir[i] == None else self.load_annotations_seq(self.img_dir, ".png", self.ann_dir, self.seg_map_suffix, self.split, img_prefix=flow_dir[i], frame_offset=offset)
            self.seq_flows.append(seq_flow)
        
        zero_index = frame_offset.index(0)
        assert zero_index != -1, "Need zero index"
        self.img_infos = self.seq_imgs[zero_index]
        self.zero_index = zero_index

        self.ds = "SynthiaSeq"
        self.data_type = data_type


        self.unpack_list = "train" in split