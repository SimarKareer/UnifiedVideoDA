from signal import SIG_DFL
from .builder import DATASETS
from .custom import CustomDataset
from .seqUtils import SeqUtils
from .viper import ViperDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import torch
from mmcv.parallel import DataContainer
import numpy as np
import pdb

@DATASETS.register_module()
class ViperSeqDataset(SeqUtils, ViperDataset):
    """Viper dataset with options for loading flow and neightboring frames.
    """

    def __init__(self, split, load_gt, img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', frame_offset=1, flow_dir=None, data_type="rgb", **kwargs):
        ViperDataset.__init__(
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
        for i, offset in enumerate(frame_offset):
            self.seq_imgs.append(self.load_annotations_seq(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=offset))
            seq_flow = None if flow_dir == None or flow_dir[i] == None else self.load_annotations_seq(self.img_dir, ".png", self.ann_dir, self.seg_map_suffix, self.split, img_prefix=flow_dir[i], frame_offset=offset)
            self.seq_flows.append(seq_flow)
        
        zero_index = frame_offset.index(0)
        assert zero_index != -1, "Need zero index"
        self.img_infos = self.seq_imgs[zero_index]
        self.zero_index = zero_index

        self.ds = "VIPER"
        
        self.data_type = data_type

        # self.fut_images, self.img_infos, self.flows = self.cofilter_img_infos(self.fut_images, self.img_infos, self.flows, self.img_dir, flow_dir, mandate_flow=False)


        self.unpack_list = "train" in split

        # breakpoint()
        # self.flow_dir = "/srv/share4/datasets/VIPER_Flowv2/train/flow_occ" #TODO Temporary, must fix or will give horrible error
        # self.flow_dir = "/srv/share4/datasets/VIPER_Flow/train/flow"
        self.palette_to_id = [(k, i) for i, k in enumerate(self.PALETTE)]

        viperClasses = ("unlabeled", "ambiguous", "sky","road","sidewalk","railtrack","terrain","tree","vegetation","building","infrastructure","fence","billboard","traffic light","traffic sign","mobilebarrier","firehydrant","chair","trash","trashcan","person","animal","bicycle","motorcycle","car","van","bus","truck","trailer","train","plane","boat")

        CSClasses = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

        cs_to_viper = {k: 201 for k in range(255)}
        for i, k in enumerate(CSClasses):
            if k in viperClasses:
                cs_to_viper[i] = viperClasses.index(k)
        
        viper_to_cs = {k: 201 for k in range(255)}
        for i, k in enumerate(viperClasses):
            if k in CSClasses:
                viper_to_cs[i] = CSClasses.index(k)
        self.convert_map = {"cityscapes_viper": cs_to_viper, "viper_cityscapes": viper_to_cs}
        self.label_space = "viper"
        
        # print("HRDA mapping: ", self.cs_to_viper)

    # def __getitem__(self, idx):
    #     print("ViperSeqDataset __getitem__")
    #     super().__getitem__(idx)
