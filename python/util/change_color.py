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
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms, utils
from Augmentor.Operations import Distort


class ChangeColorDataset(Dataset):
    """docstring for ImageGeometryDataset"""
    def __init__(self, image_dir, mode, category=None, part_name=None, height=256, width=256):
        super(ChangeColorDataset, self).__init__()
        
        self.image_dir = image_dir
        self.mode = mode
        self.category = category
        self.part_name = part_name
        self.height = int(height)
        self.width = int(width)
        if self.part_name is None:
            self.part_name = ''
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            self.distort_aug = Distort(probability=1, grid_height=3, grid_width=4, magnitude=0)

        self.transform = self.get_transform(self.mode, self.height*3, self.width*4, self.category)
        
        self.files = []
        no_patch_files = list(
                            set(glob.glob(os.path.join(self.image_dir, '*.png'))) - set(glob.glob(os.path.join(self.image_dir, '*patch*.png')))
                        )
        no_patch_files = [filename for filename in no_patch_files if self.part_name in filename]
        self.files = no_patch_files
        self.files = sorted(self.files)
        print('model num: {}'.format(len(self.files)))

        self.H_begin = [0, 256, 256, 256, 256, 512]
        self.W_begin = [256, 0, 256, 512, 768, 256]
        

    def __len__(self):
        return len(self.files)

    def get_transform(self, mode, height, width, category):
        if category == 'car':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.5, 0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5, 0.5]

        if mode == 'train' or 'val':
            transform = transforms.Compose(
                [
                    transforms.Resize((height, width)),
                    transforms.CenterCrop((height, width)),
                    # data augmentation
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        elif mode == 'test':
            transform = transforms.Compose(
                [
                    transforms.Resize((height, width)),
                    transforms.CenterCrop((height, width)),
                    # data augmentation
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.files[idx]
        basename = os.path.basename(filename)

        image = Image.open(filename)

        if self.mode == 'train' or self.mode == 'val':
            np_image = np.array(image)
            distorted_image = np.zeros((image.size[1], image.size[0], np_image.shape[2]))
            for i in range(6):
                patch = Image.fromarray(np.uint8(np_image[self.H_begin[i]:self.H_begin[i]+256, self.W_begin[i]:self.W_begin[i]+256, :]))
                distorted_patch = self.distort_aug.perform_operation([patch])
                # distorted_image[0].save(os.path.join('.', basename))
                distorted_image[self.H_begin[i]:self.H_begin[i]+256, self.W_begin[i]:self.W_begin[i]+256, :] = np.array(distorted_patch[0])
            image = Image.fromarray(np.uint8(distorted_image))

        np_image = np.array(image)
        np_image.setflags(write=1)

        if self.category == 'car':
            image = Image.fromarray(np.uint8(np_image[:, :, 0:3]))
        else:
            np_image[:, :, 3] = np_image[:, :, 3]/255
            np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
            np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
            np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
            np_image[:, :, 3] = np_image[:, :, 3]*255
            image = Image.fromarray(np.uint8(np_image[:, :, 0:4]))
        image.save(os.path.join('./temp', basename))

        if self.transform is not None:
            image = self.transform(image)
        # save_image(image, os.path.join('.', basename))
        if self.category == 'car':
            patches = torch.zeros(6, 3, self.height, self.width)
        else:
            patches = torch.zeros(6, 4, self.height, self.width)
        basenames = []
        for i in range(6):
            patches[i, :, :, :] = image[:, self.H_begin[i]:self.H_begin[i]+256, self.W_begin[i]:self.W_begin[i]+256]
            basenames.append('{}_patch{}.png'.format(basename.split('.')[0], str(i)))
        # save_image(make_grid(patches), os.path.join('.', basename))

        return patches, basenames

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_root = './20210425_plane_pixelsnail/body/top_16/auto_texture'
    category = 'car'
    part_name = 'body'
    height = 256
    width = 256
    parallel = 'False'
    mode = 'val'
    batch_size = 6

    dataset = ChangeColorDataset(
                    data_root, 
                    mode, 
                    category=category, 
                    part_name=part_name, 
                    height=height, 
                    width=width)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, (img, filename) in enumerate(dataloader):
        print(img.shape)
        from einops import rearrange, reduce, repeat
        img = rearrange(img, 'B P C H W -> (B P) C H W')
        print(img.shape)