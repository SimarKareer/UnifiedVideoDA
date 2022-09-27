from .builder import DATASETS
from .custom import CustomDataset
from .cityscapes_seq import CityscapesSeqDataset

@DATASETS.register_module()
class ViperDataset(CustomDataset):
    """Viper dataset.
    """
    #grab the classes and coloring from cityscapes dataset
    CLASSES = CityscapesSeqDataset.CLASSES
    PALETTE = CityscapesSeqDataset.PALETTE

    def __init__(self,split, **kwargs):
        super(ViperDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            split=split,
            **kwargs)