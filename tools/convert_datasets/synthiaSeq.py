# Obtained from: https://github.com/lhoyer/DAFormer

import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    # PIL does not work with the image format, but cv2 does
    label = cv2.imread(file, cv2.IMREAD_UNCHANGED)[:, :, -1]
    # mapping based on README.txt from SYNTHIA_RAND_CITYSCAPES
    id_to_trainid = {
        3: 0, # road
        4: 1, # sidewalk
        2: 2, # building 
        40: 3, # wall NOT in synthia
        5: 4, # fence
        7: 5, # pole
        15: 6, # traffic light 
        9: 7, # traffic sign
        6: 8, # vegetation 
        41: 9,  # terrain NOT in synthia
        1: 10, #sky
        10: 11, # person
        11: 12, # map bike --> rider 
        8: 13, # car
        43: 14,  # truck NOT in synthia
        44: 15, # bus NOT in synthia
        45: 16,  # train NOT  in synthia
        46: 17, #  motorcycle NOT in synthia
        47: 18 # bicycle (not used in conversion)
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds_updated.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SYNTHIA annotations to TrainIds')
    parser.add_argument('synthia_path', help='gta data path')
    parser.add_argument('--gt-dir', default='GT/LABELS/Stereo_Left/Omni_F', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    synthia_path = args.synthia_path
    out_dir = args.out_dir if args.out_dir else synthia_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(synthia_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    only_postprocessing = False

    print(len(poly_files))
    if not only_postprocessing :
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
