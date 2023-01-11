import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from tqdm import tqdm

class NpyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        
    def __getitem__(self, index):
        path1 = os.path.join(self.root, f"result/result{index}.npy")
        path2 = os.path.join(self.root, f"result_tk/result_tk{index}.npy")
        path3 = os.path.join(self.root, f"result_t_tk/result_t_tk{index}.npy")
        print(path1, path2, path3)

        result = torch.from_numpy(np.load(path1))
        result_tk = torch.from_numpy(np.load(path2))
        result_t_tk = torch.from_numpy(np.load(path3))
        return result, result_tk, result_t_tk
    
    def __len__(self):
        path1 = os.path.join(self.root, f"result")

        return len([entry for entry in os.listdir(path1) if os.path.isfile(os.path.join(path1, entry))])
        # return len(self.image_paths)

def main():
    dataset = NpyDataset('/coc/testnvme/skareer6/Projects/VideoDA/mmsegmentation/work_dirs/sourceModelCache2/')

    dataloader = torch.utils.data.DataLoader(dataset)

    for r1, r2, r3 in tqdm(dataloader):
        # print()
        breakpoint()



main()