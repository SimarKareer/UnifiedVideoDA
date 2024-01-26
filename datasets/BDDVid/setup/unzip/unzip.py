import zipfile
import glob
import os

split="test"

video_dir="" # specify directory of where videos are stored
files  = glob.glob(f"{video_dir}/videos/*.zip")

# print(files)

out_name = f"completed_{split}_unzip.txt"

if os.path.exists(out_name):
    with open(out_name, "r") as f:
        completed_zips = f.readlines()
        completed_zips = [zip.strip() for zip in completed_zips]
    completed_zips = []
else:
    completed_zips = []
completed_zips = set(completed_zips)
out_file = open(out_name, "a")


files.sort()


for file in files:
    if file in completed_zips or split not in file:
        continue
    file_id = file.split('_')[-1][:-4]
    print(file, file_id)
    with zipfile.ZipFile(file, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = os.path.basename(zip_info.filename)

            # edit output directory for where zips should be unziped
            zip_ref.extract(zip_info, f"/coc/flash9/datasets/bdd100k/bdd100k/videos/{split}/{file_id}")
    out_file.write(file + "\n")
