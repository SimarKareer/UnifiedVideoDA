import shutil
import os

split = "train"

#Edit directory for where orignal bdd10k dataset is located
orig_dataset = "/srv/datasets/bdd100k/bdd100k/images/10k/train"

# train image list 
img_list = f"../../video_da/splits/valid_imgs_{split}.txt"

# 2 options based on split: train_orig_10k or val_orig_10k
out_dataset = f"../../video_da/{split}_orig_10k/images"


if not os.path.exists(out_dataset):
    os.makedirs(out_dataset)
f = open(img_list, 'r')
files = f.readlines()
files = [x.strip() for x in files]

# get 307 frame from orig bdd10k dataset
fails = []
for file in files:
    ids = file[:-4]
    print(ids)
    orig_file = os.path.join(orig_dataset, ids + ".jpg")
    out_file = os.path.join(out_dataset, ids + "_307.jpg")
    print(orig_file, out_file)

    if not os.path.exists(orig_file):
        fails.append(file)
        continue

    os.symlink(orig_file, out_file)

print("FAILS LIST", len(fails))


# grab all other image frames (other than 307)
in_dir = f"../../video_da/{split}/images"


frame_diff = 4
for file in files:
    ids = file[:-4]

    for frame_num in range(307-frame_diff, 307 + frame_diff + 1, 1):
        orig_file = os.path.join(in_dir, ids + f"_{frame_num}.jpg")
        out_file = os.path.join(out_dataset, ids + f"_{frame_num}.jpg")
        os.symlink(orig_file, out_file)

