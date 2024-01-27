import wget
import os
import argparse

parser = argparse.ArgumentParser(description="Process a file with a given path")

# Add an argument for the file path
parser.add_argument("--file_name", required=True,type=str, help="Path to the file to process")

args = parser.parse_args()
# run it from this directory

file_name = args.file_name
file = open(file_name, "r")
links =file.readlines()

links = [link.strip() for link in links]
print("LINKS IN FILE", len(links))


out_name = f"completed_{file_name}"

if os.path.exists(out_name):
    with open(out_name, "r") as f:
        completed_zips = f.readlines()
        completed_zips = [zip.strip() for zip in completed_zips]
else:
    completed_zips = []
completed_zips = set(completed_zips)
out_file = open(out_name, "a")

#dataset out_dir
out_dir = "/coc/flash9/datasets/bdd100k/zips/videos"
for link in links:
    link_name = link.split('/')[-1]
    if link in completed_zips or os.path.exists(os.path.join(out_dir, link_name)):
        continue
    else:
        wget.download(link, out=out_dir)
        out_file.write(link + "\n")

