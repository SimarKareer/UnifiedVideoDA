import cv2
import glob
import numpy as np
import os

split = "train"

file_list = f"/coc/flash9/datasets/bdd100k/splits/video_path_{split}.txt"

out_dir = f"mmseg/datasets/BDDVid/video-da/{split}/images/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def check_best_frame(video_path):
    

    video_id = vid_path.split('/')[-1][:-4]

    file_numbers = [x for x in range(303, 312)]
    img_names = [f"{video_id}_{x}.jpg" for x in file_numbers]

    # if script fails in middle, this is to prevent us from unpacking videos again
    if os.path.exists(os.path.join(out_dir, f"{video_id}_{307}.jpg")):
        return

    vidcap = cv2.VideoCapture(video_path)
    success,img = vidcap.read()

    count = 0
    # always take frame 307 (t), 306,305,308,309
    arr = []
    while success:
        if count >= 303 and count <= 311:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            print(img.shape)
            arr.append(img)
        success,img = vidcap.read()
        count += 1
    
    if len(arr) < len(file_numbers):
        print("FAIL", video_path)
        return video_id
    for i,img_name in enumerate(img_names):
        cv2.imwrite(os.path.join(out_dir, img_name), arr[i])
    print("FINISHED", video_path)
    
    return None


f = open(file_list, 'r')
paths = f.readlines()
paths = [x.strip() for x in paths]
print(paths)

fail_list = []
for vid_path in paths:
    a = check_best_frame(vid_path)
    if a is not None:
        fail_list.append(a)

print(fail_list)
    