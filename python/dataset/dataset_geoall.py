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
    'car': [3.14917, (- 3.1562), 21.0342, (- 11.9353)], 
    'car_new_reg': [3.14917, (- 3.1562), 21.0342, (- 11.9353)], 
    'chair': [3.14224, (- 3.14214), 14.3377, (- 6.70715)], 
    'plane': [3.14295, (- 3.1416), 15.4599, (- 5.63864)], 
    'table': [3.14515, (- 3.14813), 11.043, (- 4.80372)]
    }

class GeometryAllPartsDataset(Dataset):
    """docstring for GeometryAllPartsDataset"""
    def __init__(self, mat_dir, part_names, vertex_num, mode):
        super(GeometryAllPartsDataset, self).__init__()
        self.mat_dir = mat_dir
        self.mode = mode
        self.vertex_num = vertex_num
        self.part_names = part_names
        self.category = os.path.basename(self.mat_dir)
        self.LOGR_S_MAX_MIN = category_val_dict[self.category]
        self.folders = np.loadtxt(os.path.join(self.mat_dir, self.mode+'.lst'), dtype=str)

        self.folders = sorted(self.folders)
        
    def __len__(self):
        return len(self.folders)

    def normalize(self, logr_part, s_part):
        (logrmax, logrmin, smax, smin) = self.LOGR_S_MAX_MIN
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

        cur_id = os.path.basename(self.folders[idx])
        cur_dir = os.path.dirname(self.folders[idx])
        
        origin_geo_inputs = []
        geo_inputs = []
        logrmaxs = []
        logrmins = []
        smaxs = []
        smins = []
        for part_name in self.part_names:
            fullname = os.path.join(cur_dir, cur_id, cur_id+'_'+part_name+'.mat')
            if not os.path.exists(fullname):
                LOGR = np.zeros((self.vertex_num, 3))
                S = np.zeros((self.vertex_num, 6))
            else:    
                geo_data = sio.loadmat(fullname, verify_compressed_data_integrity=False)
                LOGR = geo_data['fmlogdr']
                S = geo_data['fms']

            if LOGR.shape[0] == 1:
                LOGR = np.squeeze(LOGR, axis=0)
            if S.shape[0] == 1:
                S = np.squeeze(S, axis=0)
            origin_geo_input = np.concatenate((LOGR, S), axis = 1)
            
            LOGR, S, logrmax, logrmin, smax, smin = self.normalize(LOGR, S)
            geo_input = np.concatenate((LOGR, S), axis = 1)

            geo_inputs.append(geo_input)
            origin_geo_inputs.append(origin_geo_input)
            logrmaxs.append(logrmax)
            logrmins.append(logrmin)
            smaxs.append(smax)
            smins.append(smin)
        geo_inputs = np.array(geo_inputs)
        origin_geo_inputs = np.array(origin_geo_inputs)
        logrmaxs = np.array(logrmaxs)
        logrmins = np.array(logrmins)
        smaxs = np.array(smaxs)
        smins = np.array(smins)
        return geo_inputs, origin_geo_inputs, logrmaxs, logrmins, smaxs, smins, cur_id

if __name__ == '__main__':
    part_names = ['surface', 'left_leg1', 'left_leg2', 'left_leg3', 'left_leg4', 'right_leg1', 'right_leg2', 'right_leg3', 'right_leg4']
    dataset = GeometryAllPartsDataset(mat_dir='/mnt/f/wutong/data/table', part_names=part_names, vertex_num=2168, mode='train')
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=True)
    for b, data in enumerate(dataloader):
        print(b)
        pass
