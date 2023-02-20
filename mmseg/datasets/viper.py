from .builder import DATASETS
from .custom import CustomDataset
from .cityscapes import CityscapesDataset

@DATASETS.register_module()
class ViperDataset(CustomDataset):
    """Viper dataset.
    """
    #grab the classes and coloring from cityscapes dataset
    # CLASSES = CityscapesSeqDataset.CLASSES
    # PALETTE = CityscapesSeqDataset.PALETTE



    def __init__(self, split, img_suffix, seg_map_suffix, **kwargs):
        self.CLASSES = ("unlabeled", "ambiguous", "sky","road","sidewalk","railtrack","terrain","tree","vegetation","building","infrastructure","fence","billboard","trafficlight","trafficsign","mobilebarrier","firehydrant","chair","trash","trashcan","person","animal","bicycle","motorcycle","car","van","bus","truck","trailer","train","plane","boat")

        self.PALETTE = [[0,0,0], [111,74,0], [70,130,180], [128,64,128], [244,35,232], [230,150,140], [152,251,152], [87,182,35], [35,142,35], [70,70,70], [153,153,153], [190,153,153], [150,20,20], [250,170,30], [220,220,0], [180,180,100], [173,153,153], [168,153,153], [81,0,21], [81,0,81], [220,20,60], [255,0,0], [119,11,32], [0,0,230], [0,0,142], [0,80,100], [0,60,100], [0,0,70], [0,0,90], [0,80,100], [0,100,100], [50,0,90]]

        # breakpoint()
        if "Train" in seg_map_suffix:
            self.CLASSES = CityscapesDataset.CLASSES
            self.PALETTE = CityscapesDataset.PALETTE

        super(ViperDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            split=split,
            **kwargs)