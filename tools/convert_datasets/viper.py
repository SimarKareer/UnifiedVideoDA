# Obtained from: https://github.com/lhoyer/DAFormer

import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    #viper_palette 
    PALETTE = [[0,0,0], [111,74,0], [70,130,180], [128,64,128], [244,35,232], [230,150,140], [152,251,152], [87,182,35], [35,142,35], [70,70,70], [153,153,153], [190,153,153], [150,20,20], [250,170,30], [220,220,0], [180,180,100], [173,153,153], [168,153,153], [81,0,21], [81,0,81], [220,20,60], [255,0,0], [119,11,32], [0,0,230], [0,0,142], [0,80,100], [0,60,100], [0,0,70], [0,0,90], [0,80,100], [0,100,100], [50,0,90]]

    p = np.asarray(PALETTE)
    palette_dict = {}
    for i,x in enumerate(PALETTE):
        palette_dict[(x[2], x[1], x[0])] = i
    
    
    # re-assign labels to match the format of Cityscapes-SEQ
    # PIL does not work with the image format, but cv2 does
    label = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # mapping based on README.txt from VIPER to CITYSCAPES-SEQ

    # 15 classes to consider
    id_to_trainid = {
        3: 0, #road 
        4: 1, # sidewalk
        9: 2, # building
        40: 3, # not in viper (wall)
        11: 4, #fence 
        41: 5, # not in viper (pole)
        13: 6, # traffic light
        14: 7, # traffic sign
        7: 8, # tree to vegetation
        8: 8, # vegetation
        6: 9, #terrain
        2: 10, #sky
        20: 11, #person
        44: 12, # not in viper (rider)
        24: 13, # car
        27: 14, #truck
        26: 15, # bus
        29: 16, # train
        23: 17,  # motorcycle
        22: 18, # bicycle
    }


    sample_class_stats = {}

    label_copy = 255 * np.ones(label.shape[:2], dtype=np.uint8)

    for k,v in palette_dict.items():
        k_list = list(k)
        if v not in id_to_trainid:
            continue
        v2 = id_to_trainid[v]
        #get locations where all valuess in a row are true
        k_mask = np.sum(label == k_list, axis=2)
        
        k_mask = k_mask == 3

        label_copy[k_mask] = v2


        n = int(np.sum(k_mask))
        if n > 0:
            if v2 not in sample_class_stats:
                sample_class_stats[v2] = 0
            sample_class_stats[v2] += n

    
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VIPER annotations to TrainIds')
    parser.add_argument('viper_path', help='viper data path')
    parser.add_argument('--gt-dir', default='train/cls/', type=str)
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
    viper_path = args.viper_path
    out_dir = args.out_dir if args.out_dir else viper_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(viper_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=f'{0}.png',
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)
    print(len(poly_files))

    only_postprocessing = False
    if not only_postprocessing:
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