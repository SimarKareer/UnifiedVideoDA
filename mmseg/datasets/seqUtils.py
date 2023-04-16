import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import numpy as np
import torch
from mmcv.parallel import DataContainer
from .builder import DATASETS
import os
import flow_vis
from torchvision import transforms
from tools.aggregate_flows.flow.my_utils import flow_to_grayscale
import copy

@DATASETS.register_module()
class SeqUtils():
    def cofilter_img_infos(self, infos1, infos2, infos3, img_dir, flow_dir, mandate_flow=True):
        """
        filter infos1 and infos2 such that for each info1, info2 pair, both exist on the file system
        """
        filtered_infos1 = []
        filtered_infos2 = []
        filtered_infos3 = []
        missing_img_count = 0
        for info1, info2, info3 in zip(infos1, infos2, infos3):
            path1 = os.path.join(img_dir, info1['filename'])
            path2 = os.path.join(img_dir, info2['filename'])
            path3 = os.path.join(flow_dir, info3['filename'])
            if (mandate_flow and os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3)) or (not mandate_flow and os.path.exists(path1) and os.path.exists(path2)):
                filtered_infos1.append(info1)
                filtered_infos2.append(info2)
                filtered_infos3.append(info3)
            else:
                missing_img_count += 1
                # print("WARNING: {path1} or {path2} or {path3} does not exist")
        print(f"WARNING: {missing_img_count} images were missing from the dataset {img_dir} or {flow_dir}")
        return filtered_infos1, filtered_infos2, filtered_infos3


    def load_annotations_seq(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split, frame_offset=0):
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
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    number_length = len(img_name.split('_')[-1])
                    # img_name = f"{img_name.split('_')[0]}_{int(img_name.split('_')[1]) - frame_offset:05d}"
                    name_split = img_name.split('_')
                    num = int(name_split[-1]) - frame_offset
                    prefix = "_".join(name_split[:-1])
                    #deals with reading split files with no '_' in img names
                    if prefix == '':
                        img_name = f"{'{num:0{number_length}}'.format(num=num, number_length=number_length)}"
                    else:
                        img_name = f"{prefix}_{'{num:0{number_length}}'.format(num=num, number_length=number_length)}"
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
            # imt_imtk_flow = self.prepare_test_img(self.img_infos, idx)
            assert(self.fut_images is not None)
            assert(self.flows is not None)
            imt_imtk_flow = self.prepare_train_img(self.img_infos, idx, im_tk_infos=self.fut_images, flow_infos=self.flows)

            # im_tk = self.prepare_test_img(self.past_images, idx)
            # for k, v in im_tk.items():
            #     im_t[k+"_tk"] = v
        else:
            # im_tk_infos = self.past_images.copy()
            # im_tk_infos["prefix"] = self.flow_dir
            # print("Train mode")
            if self.flows is None:
                # print("flow off")
                imt_imtk_flow = self.prepare_train_img_no_flow(self.img_infos, idx, im_tk_infos=self.fut_images)
            else:
                # print("flow on")
                try:
                    imt_imtk_flow = self.prepare_train_img(self.img_infos, idx, im_tk_infos=self.fut_images, flow_infos=self.flows, aug_view=self.aug_view)
                except FileNotFoundError as e:
                    if self.no_crash_dataset:
                        print("Skipping image due to error: ", e)
                        imt_imtk_flow = dict(failed=True)
                    else:
                        raise e

            # im_tk = self.prepare_train_img(self.past_images, idx)
            # for k, v in im_tk.items():
            #     im_t[k+"_tk"] = v

        # if self.use_flow:

            
        
        return imt_imtk_flow
    
    def merge(self, ims, imtk, flows=None, ims_aug_1=None, ims_aug_2=None, aug_view=False):
        # print("merge input: ", ims["img"].data, imtk["img"].data, flows["flow"].data)
        # print("merge input: ", type(ims["img"]), type(imtk["img"]), type(flows["flow"]))
        # print("merge input: ", ims["img"].shape, imtk["img"].shape, flows["flow"].shape)
        if flows is None:
            ims["img"] = np.concatenate(
                (ims["img"], imtk["img"]), axis=2
            )
        else:
            if aug_view:
                ims["img"] = np.concatenate(
                    (ims["img"], imtk["img"], flows["flow"], ims_aug_1["img"], ims_aug_2["img"]), axis=2
                )
            else:
                ims["img"] = np.concatenate(
                    (ims["img"], imtk["img"] flows["flow"]), axis=2
                )

        return ims
    
    def unmerge(self, merged, aug_view=False):
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
            flows["img"] = merged["img"][:, :, 6:8]
        else:
            flows = None
        
        if aug_view:
            ims_aug_1 = merged["img"][:, :, 8:11]
            ims_aug_2 = merged["img"][:, :, 12:]
        else:
            ims_aug_1 = None
            ims_aug_2 = None

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
        assert False, "Broken after I made viper and cityscapes consistent"
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

    def prepare_train_img(self, infos, idx, im_tk_infos=None, flow_infos=None, load_tk_gt=False, aug_view=False):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
                keys: ['img_metas', 'img', 'gt_semantic_seg', 'flow', 'imtk', 'imtk_gt_semantic_seg']
                img_metas: dict
                img: [Tensor(B, C, H, W)] Y
                gt_semantic_seg: [Tensor(B, C, H, W)]
        """

        results = dict(img_info=infos[idx], ann_info=self.get_ann_info(infos, idx))
        if im_tk_infos is not None: #Bc we don't want to overwrite the image loading pipeline, we'll separately load im, imtk, flow
            resultsImtk = dict(img_info = im_tk_infos[idx], ann_info=self.get_ann_info(im_tk_infos, idx))
        if flow_infos is not None:
            resultsFlow = dict(flow_info = flow_infos[idx])

        self.pre_pipeline(results)
        self.pre_pipeline(resultsImtk)
        self.pre_pipeline_flow(resultsFlow)

        ims = self.pipeline["im_load_pipeline"](results)


        if load_tk_gt:
            imtk = self.pipeline["im_load_pipeline"](resultsImtk)
            imtk_gt = DataContainer(torch.from_numpy(imtk["gt_semantic_seg"][None, None, :, :]))
        else:
            imtk = self.pipeline["load_no_ann_pipeline"](resultsImtk)
            imtk_gt = None
        
        flows = self.pipeline["load_flow_pipeline"](resultsFlow)

        if aug_view:
            ims_aug_1 = copy.deepcopy(ims)
            ims_aug_2 = copy.deepcopy(ims)
            ImsAndFlows = self.merge(ims, imtk, flows, ims_aug_1, ims_aug_2,aug_view=aug_view) #TODO: concat the ims and flows
            ImsAndFlows = self.pipeline["shared_pipeline"](ImsAndFlows) #Apply the spatial aug to concatted im/flow
            im, imtk, flows, im_aug_1, im_aug2 = self.unmerge(ImsAndFlows, aug_view=aug_view) # separate out the ims and flows again
        else:
            ImsAndFlows = self.merge(ims, imtk, flows) #TODO: concat the ims and flows
            ImsAndFlows = self.pipeline["shared_pipeline"](ImsAndFlows) #Apply the spatial aug to concatted im/flow
            im, imtk, flows = self.unmerge(ImsAndFlows) # separate out the ims and flows again

        if ImsAndFlows["flip"]:
            flows["img"][:, :, 0] = -flows["img"][:, :, 0]

        #generate augmentations for aug views
        if aug_view:
            im_aug_1 = self.pipeline["aug_pipeline"](im_aug_1)
            im_aug_2 = self.pipeline["aug_pipeline"](im_aug_2)

            finalImsAug1 = self.pipeline["im_pipeline"](im_aug_1) #add the rest of the image augs
            finalImsAug2 = self.pipeline["im_pipeline"](im_aug_2) #add the rest of the image augs

            finalIms["im_aug_1"] = finalImsAug1["img"]
            finalIms["im_aug_2"] = finalImsAug2["img"]

        #rest of augs
        finalIms = self.pipeline["im_pipeline"](im) #add the rest of the image augs
        finalImtk = self.pipeline["im_pipeline"](imtk) #add the rest of the image augs

        finalFlows = self.pipeline["flow_pipeline"](flows) #add the rest of the flow augs

        finalIms["flow"] = finalFlows["img"]
        finalIms["imtk"] = finalImtk["img"]
        finalIms["imtk_gt_semantic_seg"] = imtk_gt

        for k, v in finalIms.items():
            if isinstance(v, DataContainer):
                finalIms[k] = v.data

        for k, v in finalIms.items():
            if isinstance(v, torch.Tensor):
                finalIms[k] = [v]

        for k, v in finalIms.items():
            if isinstance(v, np.ndarray):
                finalIms[k] = [torch.from_numpy(v)]

        def get_metas(loaded_images):
            img_metas = {}
            for k, v in loaded_images.items():
                if k not in ["img", "gt_semantic_seg", "img_info", "ann_info", "seg_prefix", "img_prefix", "seg_fields"]:
                    img_metas[k] = v

            img_metas["img_shape"] = (1080, 1920, 3)
            img_metas["pad_shape"] = (1080, 1920, 3)
            return img_metas
        finalIms["img_metas"] = [DataContainer(get_metas(ims), cpu_only=True)]
        finalIms["imtk_metas"] = [DataContainer(get_metas(imtk), cpu_only=True)]

        if load_tk_gt:
            finalIms["imtk_gt_semantic_seg"][0] = finalIms["imtk_gt_semantic_seg"][0].squeeze(0).long() #TODO, I shouldn't have to do this manually
        else:
            finalIms["imtk_gt_semantic_seg"] = [torch.tensor([])]

        finalIms["gt_semantic_seg"][0] = finalIms["gt_semantic_seg"][0].unsqueeze(0).long() #TODO, I shouldn't have to do this manually

        # Get rid of list dim for all tensors
        if self.unpack_list: #NOTE: eventually we should just always unpack the list and account for the difference in the test function.
            for k, v in finalIms.items():
                if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    finalIms[k] = v[0]
                if k == "img_metas" or k == "imtk_metas":
                    finalIms[k] = v[0]

        def simple_norm(x, mu, std):
            return (x - mu) / std

        def get_vis_flow(flow):
            # two other options are flowvis.flow_to_color, flow_to_grayscale
            is_list = isinstance(flow, list)
            if is_list:
                flow = flow[0]

            visflow = flow.norm(dim=0, keepdim=True)
            mu_of = visflow.float().mean(dim=[1, 2])
            std_of = visflow.float().std(dim=[1, 2])
            visflow = simple_norm(visflow.float(), mu_of, std_of)

            return [visflow] if is_list else visflow

        if self.data_type == "flow":
            visflow = get_vis_flow(finalIms["flow"])
            finalIms["img"] = visflow
        elif self.data_type == "rgb+flow":
            visflow = get_vis_flow(finalIms["flow"])
            finalIms["flowVis"] = visflow
            # finalIms["img"] = torch.cat([finalIms["img"], visflow], dim=0)


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