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


    def load_annotations_seq(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split, img_prefix="", frame_offset=0):
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
                    if "/" not in img_name and ".png" in img_name: #00001.png
                        img_info = dict(
                            filename=img_name, ann=dict(seg_map=img_name.split(".")[0]+seg_map_suffix)
                        )
                        img_infos.append(img_info)
                        continue

                    number_length = len(img_name.split('_')[-1])
                    # img_name = f"{img_name.split('_')[0]}_{int(img_name.split('_')[1]) - frame_offset:05d}"
                    name_split = img_name.split('_')
                    num = int(name_split[-1]) + frame_offset
                    prefix = "_".join(name_split[:-1])
                    #deals with reading split files with no '_' in img names
                    if prefix == '':
                        img_name = f"{'{num:0{number_length}}'.format(num=num, number_length=number_length)}"
                    else:
                        img_name = f"{prefix}_{'{num:0{number_length}}'.format(num=num, number_length=number_length)}"
                    img_info = dict(filename=os.path.join(img_prefix, img_name) + img_suffix)
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
        try:
            # if self.flows is None:
            #     imt_imtk_flow = self.prepare_train_img_singular(self.img_infos, idx)
            # else:
            #     assert(self.fut_images is not None)
            #     assert(self.flows is not None)
            #     imt_imtk_flow = self.prepare_train_img(self.img_infos, idx, im_tk_infos=self.fut_images, flow_infos=self.flows)
            imt_imtk_flow = self.prepare_train_img(self.seq_imgs, idx, self.seq_flows)
        except FileNotFoundError as e:
            if self.no_crash_dataset:
                print("Skipping image due to error: ", e)
                if self.flows is None:
                    imt_imtk_flow = self.prepare_train_img_singular(self.img_infos, 0)
                else:
                    imt_imtk_flow = self.prepare_train_img(self.img_infos, 0, im_tk_infos=self.fut_images, flow_infos=self.flows)
            else:
                raise e

        return imt_imtk_flow
    
    def var_merge(self, ims, flows):
        ims[self.zero_index]["img"] = np.concatenate(
            (
                [i["img"] for i in ims] + [f["flow"] for f in flows]
            ), axis=2
        )

    def var_unmerge(self, ims, flows, mergedImsFlows):
        # add assert

        idx = 0
        for i in range(len(ims)):
            ims[i]["img"] = mergedImsFlows[:, :, idx:idx+3]
            idx += 3
        
        for i in range(len(flows)):
            flows[i]["flow"] = mergedImsFlows[:, :, idx:idx+2]
            idx += 2
        
        return ims, flows

    def update_shared_metas(self, ims):
        for i in range(len(ims)):
            if i != self.zero_index:
                update_keys = ['flip_direction', 'keep_ratio', 'flip', 'scale', 'scale_idx']
                for k in update_keys:
                    ims[i][k] = ims[self.zero_index][k]
                ims[i]["gt_semantic_seg"] = torch.tensor([])
                if self.ds == "cityscapes-seq" and "valid_pseudo_mask" in ims[self.zero_index]: # during test time there is no valid pseudo mask so ignore then
                    ims[i]["valid_pseudo_mask"] = ims[self.zero_index]["valid_pseudo_mask"]


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

    
    def prepare_train_img_singular(self, infos, idx):
        results = dict(img_info=infos[idx], ann_info=self.get_ann_info(infos, idx))

        self.pre_pipeline(results)

        ims = self.pipeline["im_load_pipeline"](results)
        ims = self.pipeline["shared_pipeline"](ims)
        finalIms = self.pipeline["im_pipeline"](ims) #add the rest of the image augs

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

        finalIms["gt_semantic_seg"][0] = finalIms["gt_semantic_seg"][0].unsqueeze(0).long() #TODO, I shouldn't have to do this manually

        # Get rid of list dim for all tensors
        # if self.test_mode: #NOTE: eventually we should just always unpack the list and account for the difference in the test function.
        for k, v in finalIms.items():
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                finalIms[k] = v[0]
            if k == "img_metas" or k == "imtk_metas":
                finalIms[k] = v[0]

        if self.test_mode:
            for k, v in finalIms.items():
                if isinstance(v, torch.Tensor):
                    finalIms[k] = [v]
                elif k == "img_metas" or k == "imtk_metas":
                    finalIms[k] = [v]

        return finalIms

    def get_metas(self, loaded_images):
        img_metas = {}
        for k, v in loaded_images.items():
            if k not in ["img", "gt_semantic_seg", "img_info", "ann_info", "seg_prefix", "img_prefix", "seg_fields"]:
                img_metas[k] = v

        # img_metas["img_shape"] = (1080, 1920, 3)
        # img_metas["pad_shape"] = (1080, 1920, 3)
        return img_metas

    def backwards_compat(self, data_out):
        # put all the bindings that I had in dacs here.
        data_out["img"] = data_out["img[0]"]
        data_out["gt_semantic_seg"] = data_out["gt_semantic_seg[0]"]
        data_out["flow"] = data_out["flow[0]"]
        data_out["imtk"] = data_out["img[-1]"]
        data_out["imtk_metas"] = data_out["img_metas"]

    def prepare_train_img(self, infos, idx, flow_infos=None):
        results = [dict(img_info=infos[i][idx], ann_info=self.get_ann_info(infos[i], idx)) for i in range(len(infos))]
        resultsFlow = [None if flow_infos[i] is None else dict(flow_info=flow_infos[i][idx]) for i in range(len(flow_infos))]

        for i in range(len(results)):
            self.pre_pipeline(results[i])
        
        ims = [] #list of dicts where each dict has the im info associated with the frame_offset list in the config
        for i in range(len(results)):
            if self.load_gt[i]:
                ims.append(self.pipeline["im_load_pipeline"](results[i]))
            else:
                ims.append(self.pipeline["load_no_ann_pipeline"](results[i]))
        

        flows = [] #Same like ims but with flows
        for i in range(len(resultsFlow)):
            if self.flow_dir and self.flow_dir[i]:
                flows.append(self.pipeline["load_flow_pipeline"](resultsFlow[i]))
            else:
                flows.append(self.pipeline["stub_flow_pipeline"](self.ds))

        # Shared Pipeline
        self.var_merge(ims, flows) # Puts the merged image on the base frame (zero index)
        imsAndFlows = self.pipeline["shared_pipeline"](ims[self.zero_index]) # a single dict where img is the merged image after transforms
        ims, flows = self.var_unmerge(ims, flows, imsAndFlows["img"]) # ims and flows are both lists of dicts
        self.update_shared_metas(ims)
    
        # Correct flows for flip aug        
        if ims[self.zero_index]["flip"]:
            for i in range(len(ims)):
                flows[i]["flow"][:, :, 0] = -flows[i]["flow"][:, :, 0]

        # Rest of image and flow pipelines
        finalIms = []
        for i in range(len(ims)):
            finalIms.append(self.pipeline["im_pipeline"](ims[i]))
        for i in range(len(flows)): #rename to img
            # if resultsFlow[i]:
            flows[i]["img"] = flows[i].pop("flow")
        finalFlows = [self.pipeline["flow_pipeline"](flows[i]) if resultsFlow[i] else dict(img=torch.tensor([])) for i in range(len(flows))] #add the rest of the flow augs
        

        # Remove list dim, fix gt_sem_seg, and add flow to data_out
        data_out = {}
        for k in ["img", "gt_semantic_seg", "valid_pseudo_mask"] if "valid_pseudo_mask" in finalIms[self.zero_index] else ["img", "gt_semantic_seg"]:
            for i, frame_name in enumerate(self.frame_offset):
                if isinstance(finalIms[i][k], np.ndarray):
                    finalIms[i][k] = torch.from_numpy(finalIms[i][k])

                data_out[f"{k}[{frame_name}]"] = finalIms[i][k]
                if k == "gt_semantic_seg":
                    data_out[f"{k}[{frame_name}]"] = data_out[f"{k}[{frame_name}]"].unsqueeze(0).long()
                
        
        for i, frame_name in enumerate(self.frame_offset):
            data_out[f"flow[{frame_name}]"] = finalFlows[i]["img"]
        

        data_out["img_metas"] = finalIms[self.zero_index]["img_metas"]

        # List dims / data containers
        for k, _ in data_out.items():
            if isinstance(data_out[k], DataContainer) and "metas" not in k:
                data_out[k] = data_out[k].data
            if isinstance(data_out[k], np.ndarray):
                data_out[k] = torch.from_numpy(data_out[k])
        
        # for BW compaibility we can add all the original key names back in data_out
        self.backwards_compat(data_out)
        
        # Put stuff in lists if it's eval time.
        if not self.unpack_list:
            for k, v in data_out.items():
                if isinstance(v, torch.Tensor):
                    data_out[k] = [v]
                elif k == "img_metas" or k == "imtk_metas":
                    data_out[k] = [v]

        # before returning change the stubbed flows so they can't be used incorrectly
        return data_out

        


    def prepare_train_img_old(self, infos, idx, im_tk_infos=None, flow_infos=None, load_tk_gt=False):
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
        ImsAndFlows = self.merge(ims, imtk, flows) #TODO: concat the ims and flows
        ImsAndFlows = self.pipeline["shared_pipeline"](ImsAndFlows) #Apply the spatial aug to concatted im/flow
        im, imtk, flows = self.unmerge(ImsAndFlows) # separate out the ims and flows again

        if ImsAndFlows["flip"]:
            flows["img"][:, :, 0] = -flows["img"][:, :, 0]
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
        # if self.unpack_list: #NOTE: eventually we should just always unpack the list and account for the difference in the test function.
        for k, v in finalIms.items():
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                finalIms[k] = v[0]
            if k == "img_metas" or k == "imtk_metas":
                finalIms[k] = v[0]

        def simple_norm(x, mu, std):
            return (x - mu) / std

        def minmax_norm(x):
            return (x - x.min()) / (x.max() - x.min())

        def check_min_max(x):
            if x.max() == 0 and x.min() == 0:
                raise FileNotFoundError
            
        def get_vis_flow(flow):
            # two other options are flowvis.flow_to_color, flow_to_grayscale
            is_list = isinstance(flow, list)
            if is_list:
                flow = flow[0]

            visflow = flow.norm(dim=0, keepdim=True)
            check_min_max(visflow)
            mu_of = visflow.float().mean(dim=[1, 2])
            std_of = visflow.float().std(dim=[1, 2])

            # visflow = simple_norm(visflow.float(), mu_of, std_of)
            visflow = minmax_norm(visflow.float())

            return [visflow] if is_list else visflow

        if self.data_type == "flow":
            visflow = get_vis_flow(finalIms["flow"])
            finalIms["img"] = visflow
        elif self.data_type == "flowrgb":
            visFlow = finalIms["flow"]
            check_min_max(visFlow)
            visFlow = flow_vis.flow_to_color(visFlow.permute(1, 2, 0).numpy(), convert_to_bgr=False).transpose([2, 0, 1])
            finalIms["img"] = minmax_norm(visFlow.astype('float32'))
        elif self.data_type == "flowrgbnorm":
            visFlow = finalIms["flow"]
            check_min_max(visFlow)
            visFlow = flow_vis.flow_to_color(visFlow.permute(1, 2, 0).numpy(), convert_to_bgr=False).transpose([2, 0, 1])
            mu_of = np.array([238.27737733, 235.72995985, 226.51926128])
            std_of = np.array([37.13001504, 38.79420189, 47.94346603])
            normTrans = transforms.Normalize(mean=mu_of, std=std_of)
            finalIms["img"] = normTrans(torch.from_numpy(visFlow.astype('float32')))
        elif self.data_type == "flowxynorm":
            visFlow = finalIms["flow"]
            check_min_max(visFlow)
            visFlow = minmax_norm(visFlow)
            visFlow = torch.cat([visFlow, get_vis_flow(finalIms["flow"])])
            finalIms["img"] = visFlow
        elif self.data_type == "rgb+flow":
            visflow = get_vis_flow(finalIms["flow"])
            finalIms["flowVis"] = visflow
            # finalIms["img"] = torch.cat([finalIms["img"], visflow], dim=0)
        elif self.data_type == "rgb+flowrgb":
            visFlow = finalIms["flow"]
            check_min_max(visFlow)
            visFlow = flow_vis.flow_to_color(visFlow.permute(1, 2, 0).numpy(), convert_to_bgr=False).transpose([2, 0, 1])
            visFlow = torch.from_numpy(visFlow.astype('float32'))
            finalIms["flowVis"] = minmax_norm(visFlow)
        elif self.data_type == "rgb+flowrgbnorm":
            visFlow = finalIms["flow"]
            check_min_max(visFlow)
            visFlow = flow_vis.flow_to_color(visFlow.permute(1, 2, 0).numpy(), convert_to_bgr=False).transpose([2, 0, 1])
            mu_of = np.array([238.27737733, 235.72995985, 226.51926128])
            std_of = np.array([37.13001504, 38.79420189, 47.94346603])
            normTrans = transforms.Normalize(mean=mu_of, std=std_of)
            finalIms["flowVis"] = normTrans(torch.from_numpy(visFlow.astype('float32')))
        elif self.data_type == "rgb+flowxynorm":
            visFlow = finalIms["flow"]
            check_min_max(visFlow)
            visFlow = minmax_norm(visFlow)
            visFlow = torch.cat([visFlow, get_vis_flow(finalIms["flow"])])
            finalIms["flowVis"] = visFlow
        elif self.data_type == "rgb":
            pass
        else:
            raise Exception("Unknown data_type: {}".format(self.data_type))
    
        if self.test_mode:
            for k, v in finalIms.items():
                if isinstance(v, torch.Tensor):
                    finalIms[k] = [v]
                elif k == "img_metas" or k == "imtk_metas":
                    finalIms[k] = [v]

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