import os

# split = train or val
split="train"

# EDIT where your original bdd10k location is
orig_dataset= f"/coc/flash9/datasets/bdd100k/bdd100k/labels/sem_seg/masks/train"

# split file to look for images
img_file = f"../../video_da/splits/valid_imgs_{split}.txt"

#out-dir where labels will be stored
out_dir = f"../../video_da/{split}/labels"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# read all images we are using
with open(img_file, "r") as f:
    valid_imgs = f.readlines()

valid_imgs = [x.strip() for x in valid_imgs]

for label in valid_imgs:

    # all images are derived from orig bdd10k train set

    img_id = label[:-4] # remove the _307

    orig_label_path = os.path.join(orig_dataset, img_id + ".png")
    new_label_path = os.path.join(out_dir, label + ".png") # serves as label for t

    os.symlink(orig_label_path, new_label_path)

    
