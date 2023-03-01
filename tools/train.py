# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one
# - Add deterministic config flag

import argparse
import copy
import os
import os.path as osp
import sys
import time

import mmcv
import torch
from mmcv.runner import init_dist, _load_checkpoint, load_state_dict
from mmcv.runner.checkpoint import summarize_keys
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor, multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models.builder import build_train_model
from mmseg.utils import collect_env, get_root_logger
from mmseg.utils.collect_env import gen_code_archive


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--analysis', type=bool, default=False)
    parser.add_argument('--source-only2', type=bool, default=False)
    parser.add_argument('--debug-mode', type=bool, default=False)
    parser.add_argument('--pre-exp-check', type=bool, default=False)
    parser.add_argument('--auto-resume', type=bool, default=False)
    parser.add_argument('--nowandb', type=bool, default=False)
    parser.add_argument('--wandbid', type=str, default=None)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--l-warp-lambda', type=float, default=None)
    parser.add_argument('--l-mix-lambda', type=float, default=None)
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=False,
                    invert_dict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (bool): Whether to revise keys in state_dict.


    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    # breakpoint()

    print("Loaded state dict has keys: ", summarize_keys(state_dict.keys(), split=1))

    print("Target State Dict has keys: ", summarize_keys(model.state_dict().keys(), split=2))



    revise_dict = {
        "ema_backbone": "ema_model.backbone",
        "imnet_backbone": 'imnet_model.backbone',
        "decode_head": "model.decode_head",
        "imnet_decode_head": "imnet_model.decode_head",
        "backbone": "model.backbone",
        "ema_decode_head": "ema_model.decode_head",
    }
    if invert_dict:
        revise_dict = {v: k for k, v in revise_dict.items()}

    new_state_dict = state_dict
    if revise_keys:
        new_state_dict = {}
        for k, v in state_dict.items():
            for old_name, new_name in revise_dict.items():
                if old_name in k:
                    k = k.replace(old_name, new_name)
                    break
            new_state_dict[k] = v
            # state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    # load state_dict
    load_state_dict(model, new_state_dict, strict, logger)


    return checkpoint

def main(args):
    print("RUNNING TRAIN.PY")
    args = parse_args(args)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.model.train_cfg.work_dir = cfg.work_dir
    cfg.model.train_cfg.log_config = cfg.log_config
    if args.load_from is not None:
        assert False, "Not supported any more, use the python config to set load_from"
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if cfg.n_gpus is None else range(cfg.n_gpus)

    # init distributed env first, since logger depends on the dist info.
    if cfg.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(cfg.launcher, **cfg.dist_params)
    
    if args.lr is not None:
        print("Overwriting LR to ", args.lr)
        cfg.optimizer.lr = args.lr

    if args.l_warp_lambda is not None:
        print("Overwriting l_warp_lambda to ", args.l_warp_lambda)
        cfg.uda.l_warp_lambda = args.l_warp_lambda

    if args.l_mix_lambda is not None:
        print("Overwriting l_mix_lambda to ", args.l_mix_lambda)
        cfg.uda.l_mix_lambda = args.l_mix_lambda
    
    if args.nowandb:
        for i in range(len(cfg.log_config.hooks)):
            if cfg.log_config.hooks[i].type == "MMSegWandbHook":
                cfg.log_config.hooks.pop(i)
                break

    if args.debug_mode:
        cfg.uda.debug_mode = True

    if args.eval:
        print("EVAL MODE")
        cfg.runner.max_iters = 1
        cfg.evaluation.interval = 1
    
    if args.source_only2:
        cfg.uda.source_only2=True
    
    if args.auto_resume:
        potential_resume = os.path.join(cfg.work_dir, "latest.pth")
        if os.path.exists(potential_resume):
            cfg.resume_from = potential_resume
        print("AUTO RESUMING FROM ", cfg.resume_from)
    
    if args.wandbid:
        for i in range(len(cfg.log_config.hooks)):
            if cfg.log_config.hooks[i].type == "MMSegWandbHook":
                cfg.log_config.hooks[i].init_kwargs.id = args.wandbid
                break
    
    if args.pre_exp_check:
        cfg.evaluation.interval = 1
        cfg.checkpoint_config.interval = 1
        cfg.runner.max_iters = 2
        cfg.data.val.split="splits/tinyval.txt"
    

    print("FINISHED INIT DIST")

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump("config.py")
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # snapshot source code
    # gen_code_archive(cfg.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        deterministic = args.deterministic or cfg.get('deterministic')
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{deterministic}')
        set_random_seed(args.seed, deterministic=deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]

    model = build_train_model(
        cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # breakpoint()
    # model.backbone.patch_embed1.proj.weight vs backbone.patch_embed1.proj.weight
    # ema_model.backbone.patch_embed3.proj.bias vs ema_backbone.block3.34.attn.norm.bias
    # imnet_model.backbone.block3.26.norm1.weight vs imnet_backbone.block1.1.norm1.bias
    # ? vs imnet_decode_head.scale_attention.fuse_layer.bn.weight
    # if cfg.load_from:
    #     checkpoint = load_checkpoint(
    #         model,
    #         # "work_dirs/local-basic/230123_1434_viperHR2csHR_mic_hrda_s2_072ca/iter_28000.pth",
    #         cfg.load_from,
    #         map_location='cpu',
    #         revise_keys=False,
    #         invert_dict=False)
    #     print("LOADED A CHECKPOINT")
        

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main(sys.argv[1:])
