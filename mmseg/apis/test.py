# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
import os

from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from tools.aggregate_flows.flow.my_utils import labelMapToIm
import cv2
import pdb
import pickle
from tools.aggregate_flows.flow.my_utils import backpropFlowNoDup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.distributed as dist
import matplotlib.pyplot as plt
from mmseg.utils.custom_utils import three_channel_flow
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg, get_segmentation_error_vis, output_preds
import torchvision.transforms as transforms
import pickle as pkl
from PIL import Image


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name

def remap_labels(results, adaptation_map):
    """Remap labels according to the given mapping.

    Args:
        results [(ndarray)]: The labels to be remapped.
        adaptation_map (dict): The mapping dict.

    Returns:
        ndarray: The remapped labels.
    """
    for res in results:
        result_copy = res.copy()
        for old_id, new_id in adaptation_map.items():
            res[result_copy == old_id] = new_id
    
    return results

class NpyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        
    def __getitem__(self, index):
        path1 = os.path.join(self.root, f"result/result{index}.npy")
        path2 = os.path.join(self.root, f"result_tk/result_tk{index}.npy")
        path3 = os.path.join(self.root, f"result_t_tk/result_t_tk{index}.npy")
        path4 = os.path.join(self.root, f"gt_t/gt_t{index}.npy")
        path5 = os.path.join(self.root, f"gt_tk/gt_tk{index}.npy")
        # print(path1, path2, path3)

        result = torch.from_numpy(np.load(path1))
        result_tk = torch.from_numpy(np.load(path2))
        result_t_tk = torch.from_numpy(np.load(path3))
        gt_t = torch.from_numpy(np.load(path4))
        gt_tk = torch.from_numpy(np.load(path5))
        return result, result_tk, result_t_tk, gt_t, gt_tk
    
    def __len__(self):
        path1 = os.path.join(self.root, f"result")

        return len([entry for entry in os.listdir(path1) if os.path.isfile(os.path.join(path1, entry))])

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    metrics=["mIoU"],
                    sub_metrics=[],
                    label_space=None,
                    cache=False,
                    use_cache=False
    ):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
        metrics (list): which mIoU based metrics to include ["mIoU", "pred_pred", "gt_pred"]
        sub_metrics (list): ["mask_count", "correct_consis"]
        cache (str): directory to save cached predictions
        use_cache (str): use cached predictions to calculate metrics
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if label_space == None:
        print("WARNING: label_space is None, assuming cityscapes")
        label_space = "cityscapes"
    print("ASSUMING MODEL'S LABELS ARE", label_space)

    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    dataset.init_cml_metrics()
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    # cache=Falsegit 
    

    if use_cache:
        npy_dataset = NpyDataset(use_cache)
        dataloader = torch.utils.data.DataLoader(npy_dataset)

        for i, (r1, r2, r3, gt_t, gt_tk) in enumerate(tqdm(dataloader)):
            cached = {"pred": r1.squeeze(0), "pred_tk": r2.squeeze(0), "pred_t_tk": r3.squeeze(0), "gt_t": gt_t.squeeze(0), "gt_tk": gt_tk.squeeze(0), "index": i}
            dataset.pre_eval_dataloader_consis(None, None, None, None, cached=cached, metrics=metrics, sub_metrics=sub_metrics)


        return

    it = 0

    for batch_indices, data in zip(loader_indices, data_loader):
        it += 1
        result_tk = None
        logits = True if cache else False
        with torch.no_grad():
            refined_data = {"img_metas": data["img_metas"], "img": data["img"]}
            # breakpoint()
            result = model(return_loss=False, logits=logits, **refined_data)
            if cache or "pred_pred" in metrics or "M6Sanity" in metrics:
                refined_data = {"img_metas": data["img_metas"], "img": data["imtk"]}
                result_tk = model(return_loss=False, logits=logits, **refined_data)
            
            # if label_space != dataset.label_space:
            #     result = remap_labels(result, dataset.convert_map[f"{label_space}_{dataset.label_space}"])
            #     if result_tk is not None:
            #         result_tk = remap_labels(result_tk, dataset.convert_map[f"{label_space}_{dataset.label_space}"])
        
        if cache:
            if len(result) > 1:
                raise NotImplementedError("Only batch size 1 supported")
            
            path_dict = {}
            for item in ["result", "result_tk", "result_t_tk", "gt_t", "gt_tk"]:
                path_dict[item] = os.path.join(cache, item)
                mmcv.mkdir_or_exist(os.path.join(cache, item))
            
            
            # result = result[0][:, :, None] # for non logits
            # result_tk = result_tk[0][:, :, None]
            result = result.squeeze(0).transpose((1, 2, 0))
            result_tk = result_tk.squeeze(0).transpose((1, 2, 0))
            flow = data["flow"][0].squeeze(0).permute((1, 2, 0)).numpy()
            # breakpoint()

            result_t_tk = backpropFlowNoDup(flow, result)

            gt_t = data["gt_semantic_seg"][0]
            if gt_t.shape[0] == 1 and len(gt_t.shape) == 4:
                gt_t = gt_t.squeeze(0)
            
            gt_tk = data["imtk_gt_semantic_seg"][0]
            if gt_tk.shape[0] == 1 and len(gt_tk.shape) == 4:
                gt_tk = gt_tk.squeeze(0)
            
            np.save(os.path.join(path_dict["result"], f"result{batch_indices[0]}"), result)
            np.save(os.path.join(path_dict["result_tk"], f"result_tk{batch_indices[0]}"), result_tk)
            np.save(os.path.join(path_dict["result_t_tk"], f"result_t_tk{batch_indices[0]}"), result_t_tk)
            np.save(os.path.join(path_dict["gt_t"], f"gt_t{batch_indices[0]}"), gt_t)
            np.save(os.path.join(path_dict["gt_tk"], f"gt_tk{batch_indices[0]}"), gt_tk)

            # batch_size = 1
            # for _ in range(batch_size):
            prog_bar.update()
            continue
        

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)

        if metrics:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # Note: while above result is full preds, here it's just metrics
            
            # pred = torch.tensor(result[0]).view((1080, 1920, 1))
            # gt = torch.tensor(data["gt_semantic_seg"][0]).view((1080, 1920, 1)).long()
            # colored_pred = labelMapToIm(pred, dataset.palette_to_id)
            # colored_gt = labelMapToIm(gt, dataset.palette_to_id)
            # cv2.imwrite("work_dirs/ims/a.png", colored_pred.numpy().astype(np.int16))
            # cv2.imwrite("work_dirs/ims/b.png", colored_gt.numpy().astype(np.int16))

            if "gt_semantic_seg" in data: # Will run the original mmseg style eval if the dataloader doesn't provide ground truth
                result = dataset.pre_eval_dataloader_consis(result, batch_indices, data, predstk=result_tk, metrics=metrics, sub_metrics=sub_metrics)
            else:
                result = dataset.pre_eval(result, indices=batch_indices)
            
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model,
                   data_loader,
                   eval_settings,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   out_dir=None,
                   show=False):
    """Updated multi_gpu_test for multiframe / flow loaders"""

    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'
    metrics, sub_metrics, return_pixelwise_acc, return_confusion_matrix, return_logits, consis_confidence_thresh = eval_settings["metrics"], eval_settings["sub_metrics"], eval_settings["pixelwise accuracy"], eval_settings["confusion matrix"], eval_settings['return_logits'], eval_settings['consis_confidence_thresh']

    device = model.module.model.device
    print("TESTING METRICS: ", metrics)

    fig, axs = plt.subplots(
        5,
        2,
        figsize=(10, 10),
    )

    model.eval()
    results = []
    dataset = data_loader.dataset
    dataset.init_cml_metrics()
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    eval_metrics = [met for met in metrics if "PL" not in met]
    pl_metrics = [met for met in metrics if "PL" in met]

    it = 0
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            if "flowVis" in data and model.module.multimodal:
                refined_data = {
                    "img_metas": data["img_metas"], 
                    "img": [torch.cat([data["img"][0], three_channel_flow(data["flowVis"][0])], dim=1)]
                }
            else:
                refined_data = {
                    "img_metas": data["img_metas"], 
                    "img": [torch.cat((data["img[0]"][0], data["img[-1]"][0]), dim=0)],
                    "flow": data["flow[0]"][0]
                }
            result = model(return_loss=False, logits=return_logits, **refined_data)

            logit_result = None
            if return_logits:
                logit_result = result
                result = [np.argmax(logit_result[0], axis=0)]
                logit_result = [torch.from_numpy(logit_result[0]).to(device)]
                
            result[0] = torch.from_numpy(result[0]).to(device)

            result_tk = None
            logit_result_tk = None

            if (len(metrics) > 1 or metrics[0] != "mIoU")  and "imtk" in data:
                refined_data = {"img_metas": data["imtk_metas"], "img": data["imtk"]}
                result_tk = model(return_loss=False, logits=return_logits, **refined_data)

                if return_logits:
                    logit_result_tk = result_tk
                    result_tk = [np.argmax(logit_result_tk[0], axis=0)]
                    logit_result_tk = [torch.from_numpy(logit_result_tk[0]).to(device)]
                result_tk[0] = torch.from_numpy(result_tk[0]).to(device)
        
        # outputs the predcitions
        # if show:
        #     out_dir_1= "./mmseg/work_dirs/inference"

        #     # use the function in dacs for this
        #     # print(data["img_metas"][0].data[0][0]['ori_filename'])
        #     # print(data["imtk_metas"][0].data[0][0]['ori_filename'][:-4])

        #     img_parts_t = data["imtk_metas"][0].data[0][0]['ori_filename'].split('/')
        #     img_parts_tk = data["imtk_metas"][0].data[0][0]['ori_filename'].split('/')

        #     out_dir_1 = osp.join(out_dir_1, img_parts_t[0])

        #     if not os.path.exists(out_dir_1):
        #         os.makedirs(out_dir_1)

        #     name_t = img_parts_t[1][:-4] + "_t.png"
        #     name_tk = img_parts_tk[1][:-4] + "_tk.png"

        #     # print(name_t, name_tk)
        #     out_file_t = osp.join(out_dir_1, name_t)
        #     out_file_tk = osp.join(out_dir_1, name_tk)

        #     print(out_file_t , out_file_tk)

        #     pred_t = output_preds(result[0].cpu(), cmap='cityscapes')
        #     pred_t.save(out_file_t)

        #     pred_tk = output_preds(result_tk[0].cpu(), cmap='cityscapes')
        #     pred_tk.save(out_file_tk)

        if metrics:
            assert "gt_semantic_seg" in data, "Not compatible with current dataloader"

            eval_vals = dataset.pre_eval_dataloader_consis(curr_preds=result, data=data, future_preds=result_tk, metrics=eval_metrics, sub_metrics=sub_metrics, return_pixelwise_acc=return_pixelwise_acc, return_confusion_matrix=return_confusion_matrix, result_logits=logit_result, result_tk_logits=logit_result_tk, consis_confidence_thresh=consis_confidence_thresh)

            if out_dir:
                intersection, union, _,_  = eval_vals[0]
                iou = intersection / union
                car_iou = iou[13]
                label = (data["gt_semantic_seg"][0][0, 0]).cpu()
                pred_for_diff = torch.where(label == 255, 255, result[0].cpu()) 
                diff = torch.where(pred_for_diff == label, 255 , pred_for_diff) #creates a diff image for visualization 
                img_metas = data['img_metas'][0].data[0]
                subplotimg(axs[0, 0], result[0], cmap="cityscapes",title=("Car Miou: " + str(car_iou)))
                subplotimg(axs[1, 0], data["gt_semantic_seg"][0][0, 0], cmap="cityscapes")
                subplotimg(axs[2, 0], data["img"][0][0].permute(1, 2, 0))
                subplotimg(axs[3, 0], diff,cmap="cityscapes")


                out_file = os.path.join(out_dir,img_metas[0]['ori_filename'])
                directory = "/".join(out_file.split('/')[:-1])
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(out_file, dpi=300)
            
            # if it % 100 == 0:
            #     dataset.formatAllMetrics(metrics=metrics, sub_metrics=sub_metrics)


        results.extend(eval_vals)

        if rank == 0:
            batch_size = len(eval_vals) * world_size
            for _ in range(batch_size):
                prog_bar.update()

        it += 1


    
    for met in metrics:
        dataset.cml_intersect[met] = dataset.cml_intersect[met].cuda()
        dataset.cml_union[met] = dataset.cml_union[met].cuda()

        dist.all_reduce(dataset.cml_intersect[met], op=dist.ReduceOp.SUM)
        dist.all_reduce(dataset.cml_union[met], op=dist.ReduceOp.SUM)
        
        #for pixel wise accuracy
        dataset.pixelwise_correct[met] = dataset.pixelwise_correct[met].cuda()
        dataset.pixelwise_total[met] = dataset.pixelwise_total[met].cuda()

        dist.all_reduce(dataset.pixelwise_correct[met], op=dist.ReduceOp.SUM)
        dist.all_reduce(dataset.pixelwise_total[met], op=dist.ReduceOp.SUM)

        #for confusion matrix
        dataset.confusion_matrix[met] = dataset.confusion_matrix[met].cuda()
        dist.all_reduce(dataset.confusion_matrix[met], op=dist.ReduceOp.SUM)

    def dict_cpu(d1):
        return {k: v.cpu() for k, v in d1.items()}

    if rank == 0:
        all_metrics = {"cml_intersect": dict_cpu(dataset.cml_intersect),
                "cml_union": dict_cpu(dataset.cml_union),
                "pixelwise_correct": dict_cpu(dataset.pixelwise_correct),
                "pixelwise_total": dict_cpu(dataset.pixelwise_total),
                "confusion_matrix": dict_cpu(dataset.confusion_matrix)}
        with open(os.path.join(eval_settings['work_dir'], "eval_results.pkl"), 'wb') as f:
            pkl.dump(all_metrics, f)
    
    if rank == 0:
        dataset.formatAllMetrics(metrics=metrics, sub_metrics=sub_metrics)
    dataset.init_cml_metrics()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    
    return results


def multi_gpu_test_old(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if dataset.adaptation_map is not None:
                for res in result:
                    result_copy = res.copy()
                    for old_id, new_id in dataset.adaptation_map.items():
                        res[result_copy == old_id] = new_id

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results
