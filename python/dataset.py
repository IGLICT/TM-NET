import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb

import scipy.io as sio
import numpy as np
from PIL import Image
import re
import glob
CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs', 'id_t', 'id_b'])
# CodeRow = namedtuple('CodeRow', ['ID', 'id_tt', 'id_t', 'id_b'])
# CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs', 'id_tt', 'id_t', 'id_b'])
# CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs'])

def get_part_names(category):
    if category == 'chair':
        part_names = ['back', 'seat', 'leg_ver_1', 'leg_ver_2', 'leg_ver_3', 'leg_ver_4', 'hand_1', 'hand_2']
    elif category == 'knife':
        part_names = ['part1', 'part2']
    elif category == 'guitar':
        part_names = ['part1', 'part2', 'part3']
    elif category == 'cup':
        part_names = ['part1', 'part2']
    elif category == 'car':
        part_names = ['body', 'left_front_wheel', 'right_front_wheel', 'left_back_wheel', 'right_back_wheel','left_mirror','right_mirror']
    elif category == 'table':
        # part_names = ['surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2']
        part_names = ['surface', 'left_leg1', 'left_leg2', 'left_leg3', 'left_leg4', 'right_leg1', 'right_leg2', 'right_leg3', 'right_leg4']
    elif category == 'plane':
        part_names = ['body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'up_tail', 'down_tail', 'front_gear', 'left_gear', 'right_gear', 'left_engine1', 'right_engine1', 'left_engine2', 'right_engine2']
    else:
        raise Exception("Error")
    return part_names

def get_central_part_name(category):
    if category == 'chair':
        central_part_name = 'back'
    elif category == 'knife':
        central_part_name = 'part2'
    elif category == 'guitar':
        central_part_name = 'part3'
    elif category == 'cup':
        central_part_name = 'part1'
    elif category == 'car':
        central_part_name = 'body'
    elif category == 'table':
        central_part_name = 'surface'
    elif category == 'plane':
        central_part_name = 'body'
    else:
        raise Exception("Error")
    return central_part_name

class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

class LatentsDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))
            
        return row.ID, torch.from_numpy(row.id_tt), torch.from_numpy(row.id_t), torch.from_numpy(row.id_b)

class LatentsDataset2levels(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))
            
        return row.ID, torch.from_numpy(row.id_t), torch.from_numpy(row.id_b)

class LatentsDatasetWithGeo(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))
            
        return row.ID, torch.from_numpy(row.geo_zs), torch.from_numpy(row.id_tt), torch.from_numpy(row.id_t), torch.from_numpy(row.id_b)

class LatentsDatasetWithGeo2levels(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))
        geo_zs = np.zeros((1))
        geo_zs[0] = row.geo_zs
        return row.ID, torch.from_numpy(geo_zs), torch.from_numpy(row.id_t), torch.from_numpy(row.id_b)

class LatentsDatasetGeoOnly(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))
            
        return row.ID, torch.from_numpy(row.geo_zs)

class DownsampleImageDataset(Dataset):
    """docstring for ImageGeometryDataset"""
    def __init__(self, image_dir, transform=None):
        super(DownsampleImageDataset, self).__init__()
        
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = glob.glob(os.path.join(self.image_dir, '*', '*.png'))
        self.image_filenames = [filename for filename in self.image_filenames if '_patch' not in filename]
        self.image_filenames = [filename for filename in self.image_filenames if '_origin' not in filename]
        self.image_filenames = [filename for filename in self.image_filenames if '_restitch' not in filename]
        # print(self.image_filenames)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.image_filenames[idx]
        basename = os.path.basename(filename)

        image = Image.open(filename)
        np_image = np.array(image)
        np_image.setflags(write=1)
        np_image[:, :, 3] = np_image[:, :, 3]/255
        np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
        np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
        np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
        np_image[:, :, 3] = np_image[:, :, 3]*255
        # print(np_image.shape)
        image = Image.fromarray(np.uint8(np_image))
        print(np_image.shape)

        if self.transform is not None:
            image = self.transform(image)
        print(image.shape)

        return image, basename

class PatchImageDataset(Dataset):
    """docstring for ImageGeometryDataset"""
    def __init__(self, image_dir, transform=None, part_name=None):
        super(PatchImageDataset, self).__init__()
        
        self.image_dir = image_dir
        self.transform = transform
        self.part_name = part_name
        self.image_filenames = glob.glob(os.path.join(self.image_dir, '*', '*patch*.png'))
        if self.part_name is not None:
            self.image_filenames = [filename for filename in self.image_filenames if self.part_name in filename]
        self.image_filenames = sorted(self.image_filenames)
        print(self.image_filenames)
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.image_filenames[idx]
        basename = os.path.basename(filename)

        image = Image.open(filename)
        np_image = np.array(image)
        np_image.setflags(write=1)
        # np_image[:, :, 3] = np_image[:, :, 3]/255
        # np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
        # np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
        # np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
        # np_image[:, :, 3] = np_image[:, :, 3]*255
        image = Image.fromarray(np.uint8(np_image[:, :, 0:3]))

        if self.transform is not None:
            image = self.transform(image)

        return image, basename

class AllPartsnpzImageDataset(Dataset):
    """docstring for ImageGeometryDataset"""
    def __init__(self, image_dir):
        super(AllPartsnpzImageDataset, self).__init__()
        
        self.image_dir = image_dir
        self.image_filenames = glob.glob(os.path.join(self.image_dir, '*', '*_merge.npz'))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.image_filenames[idx]
        basename = os.path.basename(filename)

        f = open(filename, 'rb')
        image = np.load(f)
        f.close()
        image = image.astype('float32')
        return image, basename

class AllPartsnpzNHWCImageDataset(Dataset):
    """docstring for ImageGeometryDataset"""
    def __init__(self, image_dir):
        super(AllPartsnpzNHWCImageDataset, self).__init__()
        
        self.image_dir = image_dir
        self.image_filenames = glob.glob(os.path.join(self.image_dir, '*', '*_merge_NH_W_C.npz'))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.image_filenames[idx]
        basename = os.path.basename(filename)

        f = open(filename, 'rb')
        image = np.load(f)
        f.close()
        image = image.astype('float32')
        return image, basename

class GeometryDataset(Dataset):
    """docstring for GeometryDataset"""
    def __init__(self, mat_dir, part_name=''):
        super(GeometryDataset, self).__init__()
        
        self.mat_dir = mat_dir

        self.mat_filenames = glob.glob(os.path.join(self.mat_dir, '*', '*'+part_name+'.mat'))
        self.mat_filenames = [mat_filename for mat_filename in self.mat_filenames if 'acap' not in mat_filename]
        self.mat_filenames = sorted(self.mat_filenames)
        print(self.mat_filenames)
    
    def __len__(self):
        return len(self.mat_filenames)

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

        fullname = self.mat_filenames[idx]
        # print(fullname)
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
        
        return geo_input, origin_geo_input, logrmax, logrmin, smax, smin, fullname

class GeometryDatasetALL(Dataset):
    """docstring for GeometryDatasetALL"""
    def __init__(self, mat_dir, category='car', vertex_num=9602):
        super(GeometryDatasetALL, self).__init__()
        self.mat_dir = mat_dir
        self.category = category
        self.vertex_num = vertex_num
        self.part_names = get_part_names(self.category)

        self.ids = glob.glob(os.path.join(self.mat_dir, '*'))
        self.ids = sorted(self.ids)
        # print(self.mat_filenames)
    def __len__(self):
        return len(self.ids)

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

        cur_id = os.path.basename(self.ids[idx])
        cur_dir = os.path.dirname(self.ids[idx])
        
        origin_geo_inputs = []
        geo_inputs = []
        logrmaxs = []
        logrmins = []
        smaxs = []
        smins = []
        for part_name in self.part_names:
        # print(fullname)
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

class FeatureMapDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            row = pickle.loads(txn.get(key))

        return row.ID, torch.from_numpy(row.quant_t), torch.from_numpy(row.quant_b)