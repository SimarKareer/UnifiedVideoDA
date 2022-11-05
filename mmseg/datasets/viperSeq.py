from signal import SIG_DFL
from .builder import DATASETS
from .custom import CustomDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import torch
from mmcv.parallel.data_container import DataContainer
import numpy as np

@DATASETS.register_module()
class ViperSeqDataset(CustomDataset):
    """Viper dataset with options for loading flow and neightboring frames.
    """
    #grab the classes and coloring from cityscapes dataset
    # CLASSES = CityscapesSeqDataset.CLASSES
    # PALETTE = CityscapesSeqDataset.PALETTE

    CLASSES = ("unlabeled", "ambiguous", "sky","road","sidewalk","railtrack","terrain","tree","vegetation","building","infrastructure","fence","billboard","trafficlight","trafficsign","mobilebarrier","firehydrant","chair","trash","trashcan","person","animal","bicycle","motorcycle","car","van","bus","truck","trailer","train","plane","boat")

    PALETTE = [[0,0,0], [111,74,0], [70,130,180], [128,64,128], [244,35,232], [230,150,140], [152,251,152], [87,182,35], [35,142,35], [70,70,70], [153,153,153], [190,153,153], [150,20,20], [250,170,30], [220,220,0], [180,180,100], [173,153,153], [168,153,153], [81,0,21], [81,0,81], [220,20,60], [255,0,0], [119,11,32], [0,0,230], [0,0,142], [0,80,100], [0,60,100], [0,0,70], [0,0,90], [0,80,100], [0,100,100], [50,0,90]]

    def __init__(self, split, flow_dir=None, frame_offset=1, **kwargs):
        super(ViperSeqDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            split=split,
            **kwargs)
        
        self.flow_dir = flow_dir
        self.past_images = self.load_annotations2(self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, frame_offset=frame_offset)
        self.flows = None if self.flow_dir == None else self.load_annotations2(self.img_dir, ".png", self.ann_dir, self.seg_map_suffix, self.split, frame_offset=1)
        # self.flow_dir = "/srv/share4/datasets/VIPER_Flowv2/train/flow_occ" #TODO Temporary, must fix or will give horrible error
        # self.flow_dir = "/srv/share4/datasets/VIPER_Flow/train/flow"
    
    def load_annotations2(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split, frame_offset=0):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
            frame_distance

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        flow_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_name = f"{img_name.split('_')[0]}_{int(img_name.split('_')[1]) - frame_offset:05d}"
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            raise NotImplementedError("must specify split")
        #     for img in self.file_client.list_dir_or_file(
        #             dir_path=img_dir,
        #             list_dir=False,
        #             suffix=img_suffix,
        #             recursive=True):
        #         img_info = dict(filename=img)
        #         if ann_dir is not None:
        #             seg_map = img.replace(img_suffix, seg_map_suffix)
        #             img_info['ann'] = dict(seg_map=seg_map)
        #         img_infos.append(img_info)
        #     img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            im_t = self.prepare_test_img(self.img_infos, idx)
            # im_tk = self.prepare_test_img(self.past_images, idx)
            # for k, v in im_tk.items():
            #     im_t[k+"_tk"] = v
        else:
            # im_tk_infos = self.past_images.copy()
            # im_tk_infos["prefix"] = self.flow_dir
            if self.flows is None:
                # print("flow off")
                imt_imtk_flow = self.prepare_train_img_no_flow(self.img_infos, idx, im_tk_infos=self.past_images)
            else:
                # print("flow on")
                imt_imtk_flow = self.prepare_train_img(self.img_infos, idx, im_tk_infos=self.past_images, flow_infos=self.flows)

            # im_tk = self.prepare_train_img(self.past_images, idx)
            # for k, v in im_tk.items():
            #     im_t[k+"_tk"] = v

        # if self.use_flow:

            
        
        return imt_imtk_flow
    
    def merge(self, ims, imtk, flows=None):
        # print("merge input: ", ims["img"].data, imtk["img"].data, flows["flow"].data)
        # print("merge input: ", type(ims["img"]), type(imtk["img"]), type(flows["flow"]))
        # print("merge input: ", ims["img"].shape, imtk["img"].shape, flows["flow"].shape)
        if flows is None:
            ims["img"] = np.concatenate(
                (ims["img"], imtk["img"]), axis=2
            )
        else:
            ims["img"] = np.concatenate(
                (ims["img"], imtk["img"], flows["flow"]), axis=2
            )

        return ims
    
    def unmerge(self, merged):
        def copy_no_img(merged):
            copy = {}
            for k, v in merged.items():
                if k != "img":
                    copy[k] = v
            return copy

        imtk = copy_no_img(merged)

        # print(merged["img"].shape)
        imtk["img"] = merged["img"][:, :, 3:6]

        if self.flows is not None:
            flows = copy_no_img(merged)
            flows["img"] = merged["img"][:, :, 6:]
        else:
            flows = None
        merged["img"] = merged["img"][:, :, :3]
        # print("merge input: ", merged["img"].shape, imtk["img"].shape, flows["img"].shape)
        return merged, imtk, flows
    
    def pre_pipeline_flow(self, results):
        """Prepare results dict for pipeline."""
        # results['seg_fields'] = []
        results['flow_prefix'] = self.flow_dir
        # results['seg_prefix'] = self.ann_dir
        # if self.custom_classes:
        #     results['label_map'] = self.label_map

    
    def prepare_train_img_no_flow(self, infos, idx, im_tk_infos):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = infos[idx]
        ann_info = self.get_ann_info(infos, idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        resultsImtk = dict(img_info = im_tk_infos[idx])

        self.pre_pipeline(results)
        self.pre_pipeline(resultsImtk)

        ims = self.pipeline["im_load_pipeline"](results)
        #TODO can improve performance if we only run the im_load_pipeline when we want imtk annotations
        # imtk = self.pipeline["im_load_pipeline"](resultsImtk)
        imtk = self.pipeline["im_load_pipeline"](resultsImtk)
        imtk_gt = imtk["gt_semantic_seg"]

        mergedIms = self.merge(ims, imtk) #TODO: concat the ims and flows

        mergedIms = self.pipeline["shared_pipeline"](mergedIms) #Apply the spatial aug to concatted im/flow
        im, imtk, _ = self.unmerge(mergedIms) # separate out the ims and flows again
        finalIms = self.pipeline["im_pipeline"](im) #add the rest of the image augs
        finalImtk = self.pipeline["im_pipeline"](imtk) #add the rest of the image augs
        
        finalIms["imtk"] = finalImtk["img"]
        finalIms["imtk_gt_semantic_seg"] = imtk_gt
        return finalIms

    def prepare_train_img(self, infos, idx, im_tk_infos=None, flow_infos=None):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # img_info = infos[idx]
        # ann_info = self.get_ann_info(infos, idx)
        results = dict(img_info=infos[idx], ann_info=self.get_ann_info(infos, idx))
        # print("results: ", results)
        # print("im_tk_infos: ", im_tk_infos)
        # print("flow_infos: ", flow_infos)
        if im_tk_infos is not None: #Bc we don't want to overwrite the image loading pipeline, we'll separately load im, imtk, flow
            # resultsImtk = results.copy()
            resultsImtk = dict(img_info = im_tk_infos[idx], ann_info=self.get_ann_info(im_tk_infos, idx))
        if flow_infos is not None:
            # resultsFlow = results.copy()
            # resultsFlow["flow_info"] = flow_infos
            resultsFlow = dict(flow_info = flow_infos[idx])
        # print(resultsFlow["flow_info"])
        self.pre_pipeline(results)
        self.pre_pipeline(resultsImtk)
        self.pre_pipeline_flow(resultsFlow)

        ims = self.pipeline["im_load_pipeline"](results)
        imtk = self.pipeline["im_load_pipeline"](resultsImtk)
        imtk_gt = imtk["gt_semantic_seg"][None, None, :, :]
    
        flows = self.pipeline["load_flow_pipeline"](resultsFlow)
        # print("flows after load: ", flows["flow"], type(flows["flow"]))
        # print("ims after load: ", ims["img"], type(ims["img"]))
        ImsAndFlows = self.merge(ims, imtk, flows) #TODO: concat the ims and flows
        # print(ImsAndFlows.keys())
        ImsAndFlows = self.pipeline["shared_pipeline"](ImsAndFlows) #Apply the spatial aug to concatted im/flow
        im, imtk, flows = self.unmerge(ImsAndFlows) # separate out the ims and flows again
        finalIms = self.pipeline["im_pipeline"](im) #add the rest of the image augs
        finalImtk = self.pipeline["im_pipeline"](imtk) #add the rest of the image augs
        finalFlows = self.pipeline["flow_pipeline"](flows) #add the rest of the flow augs
        finalIms["flow"] = finalFlows["img"]
        finalIms["imtk"] = finalImtk["img"]
        finalIms["imtk_gt_semantic_seg"] = imtk_gt
        # return self.pipeline(results)
        return finalIms

    def prepare_test_img(self, infos, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def get_ann_info(self, infos, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return infos[idx]['ann']