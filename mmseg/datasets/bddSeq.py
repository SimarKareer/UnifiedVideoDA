import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset
from .cityscapesSeq import CityscapesSeqDataset
from .seqUtils import SeqUtils
import cityscapesscripts.helpers.labels as CSLabels
import pdb

@DATASETS.register_module()
class BDDSeqDataset(CityscapesSeqDataset):
    """BDD-Seq dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelIds.png' for BDD-seq dataset.
    """

    def __init__(self, split, load_gt, img_suffix='.jpg', seg_map_suffix='.png', frame_offset=-2, flow_suffix=".png", **kwargs):
        # breakpoint()

        CityscapesSeqDataset.__init__(
            self, #must explicitly pass self
            split,
            load_gt,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            frame_offset=frame_offset,
            flow_suffix=flow_suffix,
            ds="bdd",
            **kwargs)
        SeqUtils.__init__(self)