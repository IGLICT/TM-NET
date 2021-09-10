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

category_val_dict = {
    'car': [3.14917, -3.1562, 21.0342, -11.9353], 
    'car_new_reg': [3.14917, -3.1562, 21.0342, -11.9353], 
    'chair': [3.14224, -3.14214, 14.3377, -6.70715], 
    'plane': [3.14295, -3.1416, 15.4599, -5.63864], 
    'table': [3.14515, -3.14813, 11.043, -4.80372]
    }

class GeoVAEDataset(Dataset):
    """docstring for GeoVAEDataset"""
    def __init__(self, mat_dir, mode, part_name=None):
        super(GeoVAEDataset, self).__init__()
        
        self.mat_dir = mat_dir
        self.mode = mode
        self.folders = np.loadtxt(os.path.join(self.mat_dir, self.mode+'.lst'), dtype=str)
        self.category = os.path.basename(self.mat_dir)
        self.part_name = part_name
        self.LOGR_S_MAX_MIN = category_val_dict[self.category]

        if self.part_name is None:
            self.part_name = ''
        
        self.files = []
        for folder in self.folders:
            self.files = self.files + glob.glob(os.path.join(self.mat_dir, folder, '*'+self.part_name+'*.mat'))
        self.files = [file for file in self.files if 'acap' not in file]
        self.files = sorted(self.files)
        # print(self.files)
        print(len(self.files))
    
    def __len__(self):
        return len(self.files)

    def normalize(self, logr_part, s_part):
        logrmax, logrmin, smax, smin = self.LOGR_S_MAX_MIN
        resultmin = -0.95
        resultmax = 0.95
        
        logrmin = logrmin - 1e-6
        logrmax = logrmax + 1e-6
        smin = smin - 1e-6
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
        try:
            LOGR = geo_data['fmlogdr']
            S = geo_data['fms']
        except:
            print(fullname)
            return
        
        if LOGR.shape[0] == 1:
            LOGR = np.squeeze(LOGR, axis=0)
        if S.shape[0] == 1:
            S = np.squeeze(S, axis=0)
        origin_geo_input = np.concatenate((LOGR, S), axis = 1)
        
        LOGR, S, logrmax, logrmin, smax, smin = self.normalize(LOGR, S)
        geo_input = np.concatenate((LOGR, S), axis = 1)
        
        return geo_input, origin_geo_input, logrmax, logrmin, smax, smin, fullname

if __name__ == '__main__':
    dataset = GeoVAEDataset(mat_dir='/mnt/f/wutong/data/table', mode='train', part_name='left_leg1')
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=True)
    for b, data in enumerate(dataloader):
        print(b)
        pass