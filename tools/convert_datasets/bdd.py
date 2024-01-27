# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image


def convert_json_to_label(label_file):

    pil_label = Image.open(label_file)
    label = np.asarray(pil_label)
    sample_class_stats = {}
    for c in range(19):
        n = int(np.sum(label == c))
        if n > 0:
            sample_class_stats[int(c)] = n
    sample_class_stats['file'] = label_file
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert bdd annotations to TrainIds')
    parser.add_argument('bdd_path', help='bdd data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(args, out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, f'{args.split}_sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, f'{args.split}_sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, f'{args.split}_samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    bdd_path = args.bdd_path
    out_dir = args.out_dir if args.out_dir else bdd_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(bdd_path, args.split, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(gt_dir,'307.png', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, f'{args.split}_sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(args, out_dir, sample_class_stats)

if __name__ == '__main__':
    main()
