import glob
import os
import pickle
import re
from collections import namedtuple

import lmdb
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

class LatentGeo2LevelsDataset(Dataset):
    """docstring for LatentGeo2LevelsDataset"""
    def __init__(self, mat_dir, mode, part_name=None):
        super(LatentGeo2LevelsDataset, self).__init__()
        
        self.mat_dir = mat_dir
        self.mode = mode
        self.folders = np.loadtxt(os.path.join(self.mat_dir, self.mode+'.lst'), dtype=str)
        self.part_name = part_name
        if self.part_name is None:
            self.part_name = ''
        self.files = []
        for folder in self.folders:
            self.files = self.files + glob.glob(os.path.join(self.mat_dir, folder, '*'+self.part_name+'.mat'))
        self.files = [file for file in self.files if 'acap' not in file]
        self.files = sorted(self.files)
        print(len(self.files))
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fullname = self.files[idx]
        # print(fullname)
        data_dict = sio.loadmat(fullname, verify_compressed_data_integrity=False)
        geo_z = data_dict['geo_z']
        id_ts = data_dict['id_ts']
        id_bs = data_dict['id_bs']

        return geo_z, id_ts, id_bs, fullname

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_root = './car_latents'
    category = 'car'
    part_name = 'body'
    height = 256
    width = 256
    parallel = 'False'
    mode = 'train'
    batch_size = 6
    dataset = LatentGeo2LevelsDataset(
                    data_root, 
                    mode, 
                    part_name=part_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, (geo_z, id_ts, id_bs, fullname) in enumerate(dataloader):
        if torch.isnan(geo_z).any():
            print(fullname)