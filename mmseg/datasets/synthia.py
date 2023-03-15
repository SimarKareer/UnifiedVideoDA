# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SynthiaDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, split, img_suffix, seg_map_suffix, **kwargs):

        # self.CLASSES = # search online whats the classes order
        # self.PALETTE = #search online whats the palette 

        if "Train" in seg_map_suffix:
            self.CLASSES = CityscapesDataset.CLASSES
            self.PALETTE = CityscapesDataset.PALETTE

        super(SynthiaDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            split=split,
            **kwargs)
