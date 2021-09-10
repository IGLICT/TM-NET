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

class SPVAEDataset(Dataset):
    """docstring for SPVAEDataset"""
    def __init__(self, mat_dir, mode):
        super(SPVAEDataset, self).__init__()
        
        self.mat_dir = mat_dir
        self.mode = mode
        self.folders = np.loadtxt(os.path.join(self.mat_dir, self.mode+'.lst'), dtype=str)
        self.files = []
        for folder in self.folders:
            self.files = self.files + glob.glob(os.path.join(self.mat_dir, folder, 'geo_zs.mat'))
        self.files = [file for file in self.files if 'acap' not in file]
        self.files = sorted(self.files)
        # print(self.files)
        print(len(self.files))
    
    def __len__(self):
        return len(self.files)

    def normalize(self, logr_part, s_part):
        resultmin = -0.95
        resultmax = 0.95

        logrmin = logr_part.min()
        logrmin = logrmin - 1e-6
        logrmax = logr_part.max()
        logrmax = logrmax + 1e-6

        smin = s_part.min()
        smin = smin - 1e-6
        smax = s_part.max()
        smax = smax + 1e-6

        rnew = (resultmax - resultmin) * (logr_part - logrmin) / (logrmax - logrmin) + resultmin
        snew = (resultmax - resultmin) * (s_part - smin) / (smax - smin) + resultmin
        return rnew, snew, logrmax, logrmin, smax, smin

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fullname = self.files[idx]
        # print(fullname)
        geo_data = sio.loadmat(fullname, verify_compressed_data_integrity=False)
        geo_zs = geo_data['geo_zs']
        
        return geo_zs, fullname
