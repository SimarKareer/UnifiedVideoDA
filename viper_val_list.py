import os
import json 
import glob


path = "/srv/share4/datasets/VIPER/val/img/"

files = [x for x in glob.glob(path + "**/*0.jpg")]

files = [x[35:] for x in files ]
print(files)

files.sort()
print("****")
print()
print()
print(files)


f = open('val.txt', 'w')


for x in files:
    f.write(x + "\n")
f.close()


