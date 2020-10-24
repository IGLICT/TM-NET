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

class RestitchImageDataset(Dataset):
    """docstring for ImageGeometryDataset"""
    def __init__(self, image_dir, transform=None):
        super(RestitchImageDataset, self).__init__()
        
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = glob.glob(os.path.join(self.image_dir, '*', '*restitch*.png'))
        all_white_ids = ['1ab8a3b55c14a7b27eaeab1f0c9120b7', '1ac6531a337de85f2f7628d6bf38bcc4', '1c2cad4df1697103a5139737ddc33fc8', '1ee30d218c8c730ecb01bc908e8cea6', '1f1b07bf637b3a1582db9fca4b68095', '1fd7d0fe883059e355feea066cda8f9a', '1fe0a612dfc3fc548be8b77157b819c9', '2a56e3e2a6505ec492d9da2668ec34c', '2ae70fbab330779e3bff5a09107428a5', '2c7d258bd18de1c28f7b968097bdb0fd', '2c351621045e25351385913bc50a0210', '2e0beb3b6927a2b7e45ef4135c266a12', '2f0a94efe6d1da7f8616812464c86290', '2f1a67cdabe2a70c492d9da2668ec34c', '3aaa59b19eebcb5f41552c6ecbda964b', '3aab16309520fb21dc0a8cba62d9a78a', '3ab2563609476e089868adf0764f7fd9', '3af3096611c8eb363d658c402d71b967', '3b1f1913f2bc0dc171dbe96559c7bcae', '3b3a9f4e3aa9f2f4d39a194653571dfc', '3b788994cd578990c35131da26f8061a', '3bb8e6e640c32a7c36b0f2a1430e993a', '3c8362c1e57c30d7e6c5cd45aa112726', '3cf6db91f872d26c222659d33fd79709', '3d442388af50f61138ae3771b1788e76', '3dd04235c696c117db6d2ff65cc5b310', '3ea40a75f22515557dcf230d8b7d162e', '3efa8030c71fab9055afe589a9f09982', '3f194ed2680ac0f4f875094dbf5fec47', '3ffd794e5100258483bc207d8a5912e3', '4a17e5189aafa9afc8687ff9b0b4e4ac', '4a5332f5aaa3144836b0f2a1430e993a', '4bc5920f74bbf6412acd2bb6205825cb', '4c3f452cd60df0efb0957d845ac33749', '4c513ea0804fc008c8687ff9b0b4e4ac', '4dcf1f64f70f22d7b4569422e5e8ba0', '4e5c18c87a10f050e7469a3f8248f48', '4e8d8792a3a6390b36b0f2a1430e993a', '4e26eab28703c12bdd5f3f2440a93d21', '4e358c2dc0513971f98c0761af40e04', '4e9475fea6f3ca32c4cb3d6800567627', '4ed8015bdbd8c03d2f95e2a1e9997b85', '4eed9ad9465482d543b9eef399a0841d', '4fd4cda3d713bb29c8687ff9b0b4e4ac', '5a96894b5cde0551c8687ff9b0b4e4ac', '5ac8b44ff77e5490c8687ff9b0b4e4ac', '5b0e833cf2ea465e42bd82e95587d8af', '5c1c5dd72247cb6526428c7519676cd8', '5d02aed0e9c93e829b9f2eb77f5e247e', '5d3cfbc8bae19e701f580a0f9153317f', '5d20adaf6d8f89fa2f1c10544d7d6f', '5d959b0f79a22e8c67c9124d122355ab', '5e2d7d43431eea85364b7ec2e28b3bd', '5e338c489e940ab73aab636b8c7f0dd2', '5e706e87ca60bd19ecb01bc908e8cea6', '5eaa2730f10054d0f6cabe1df6f4c9d9', '5eb38d4e556a893a3b832d9b48bc2dd3', '5eb67ecc35e39eb59753496ba23f2183', '5edfec789343e0c3319f1c1eee46f332', '6a8e63f10bd6736e713decb1a0563b12', '6a8f1dd7e0642ca4b367a648e9b173ab', '6a73ce89c4a941ac0f947750540fb22', '6b9dc32e4d278663c8687ff9b0b4e4ac', '6b10b9f8b8f38cfadebf8d14d5516c20', '6b881cd27826bd6b97a79b27e081b077', '6cf4bcb8a694242473ae2443c97a1733', '6cfe2f5ab1bb89022edea1c2bfa3bc56', '6d4ce042f204e89db413bc67c6e4035a', '6dc02b1884fb578492d9da2668ec34c', '6ed05dd2e39e06223b12a1486cdc4b7', '6f6191af40bfc07116f5e76c24d541f3', '6fd76577d0df60669b9f2eb77f5e247e', '6ffbc6f14a7257caffa41024c63fa9db', '7a5d539e3771a491ca6a405498436716', '7ae007a84e6e616d5781796ac0b9d597', '7d7b3b8a1734154b3b8d5c0ebdd194c8', '7ee09fdece7d9142afdb9a672b7d3b8a', '7efeece3b5cf2853d706779c93538ee1', '8a232028c2b2cfad43649af30eba8304', '8ad5b0bd9f1297e5492d9da2668ec34c', '8ad35dbc44f40be1492d9da2668ec34c', '8b8fa92f9c677b0713decb1a0563b12', '8b016c4df0e139b660065f613e3a3cf', '8b0174afa05ed3c2dcadbd99efe6967', '8b886cf32f340c89a9e00c019d029152', '8c4d57109ce0f2f222659d33fd79709', '8cca53295cb61e878d3e8783b997c75', '8d5866194748bb3492d9da2668ec34c', '8da6959dc59296c9f2f43e6f841bd32b', '8db4d399adc4d6148738e7290b6f3237', '8e1d232ce609392ebd8eb8ad250e5135', '8e7714615a4b1e6f82390c5f604e0d9b', '8ff4ba87d700054546992ce9fde1b2c2', '9b9a114cfee79bfb492d9da2668ec34c', '9b275819a433a9c733e4fde2bc371944', '9bbccaed652a88d0ad4feb1e0e9b77d9', '9c50878c91aeb8126bb6bc0db07c71e8', '9ce08b44810f03c963cf8719fe855f3e', '9d0b25421c13008e35836c728d324152', '9d36bf414dde2f1a93a28cbb4bfc693b', '18fd8342fa5d1d4f5268b70948af88b2', '19a3ac2ec0312e13c8687ff9b0b4e4ac', '21a7166815b80f3f7fee47aa19e2e28c', '22f030ce09693f4c65c99f3aded15b93', '23e726da58f115f69b9f2eb77f5e247e', '26e8033e59a3adf6bb53a6a5f5051240', '27a8827b0ba5d743169bfe841050f4a4', '27ea798c55699b6d2c528d33bca1ac2', '32d9f69ef8ebb1778a514cac7cb18507', '32d243fc56b7b8658cbd400aa99bee96', '36a2d17d63dc83f983c05814b11dc86', '36f85b1709a62c406daafc9da8386a39', '38f8b44fc8dac88ce3c886e48eaa6e2d', '40e9fdb477fc0c10f07ea52432becd0a', '40ee8ed17f6ea51224669056e0d19a1', '41ce60d5443c203eb31c248b8665b2e7', '44aed693950b077d4f9691cdd5f8749', '45c998547946bc4d6a88c250a18c2861', '45dcb53af7bea0d091912372ddaeb001', '45f9f54b9e7cf6897886c5f50906be82', '46f6a6e0f239282fc8687ff9b0b4e4ac', '47c04fe5c1f850b784cb23b18bd34686', '48d44532ffcdd4a2fc30bf6907b3dbb9', '49aa713bec70ee1f1104b8f54582c707', '49c955a80749d2e1a5ffdf44ff86b795', '50faa1777b012058492d9da2668ec34c', '55c9891ac274e076492d9da2668ec34c', '56fc424a89bb137bf2197da8dec8488d', '57d4b5a07e67c24af77e1de7c7a7b6e7', '58b1c10a6e1d67b3b0cf2e642b746bfb', '61bc4930719717a9492d9da2668ec34c', '61fb89fca404e37f598bcf8ac674b63d', '63da17eda9d415b5319c5e90e9cc9126', '65b8c99a5ab7eb0492ea5a71c9e33093', '66aea803dc4a6c5390907a4c1a185a7', '66e204998b6c4f06ffe5cbaf6252d67f', '66f18d05d960ffe0bcd12732b5a4b789', '67fe832a22b91e6c54ddc93d2b7d3d65', '70f57047512c2eb84104b1c5cb7f9280', '72fd9c2f47db23987f377effd1f7ba9e', '75ea5a697313a8c214c2f69de20984ee', '76ee694df725b5b4664b3b9b23ddfcbc', '77a4269a0d64d48336b0f2a1430e993a', '78cf0ebb875de8dfba18ade30e563d37', '78e1977bc5f0f4041552c6ecbda964b', '79a3115a6f96eef7c151419181ef256', '80f96250bed47b4f52e9aa88ca53e944', '86e2424968b01e241b94efa300287e71', '89d2cbab56c5a1d7297056ac457c1c31', '89f8c12dafc877009f8c3d2002c77ddb', '90cae938e34b11a429823c412ca4576c', '92cae2e67b788eaa9dcc460592c0e125', '94b55dcb770626d9c8687ff9b0b4e4ac', '94ceeee26248a275e8e2378aa23e4253', '95e1571acdd75922afdb9a672b7d3b8a', '96d9944f46bd330979414e04132a8bef', '97bbc8970b05c4a3fcde6bcb709edd9a', '97cd4ed02e022ce7174150bd56e389a8', '107caefdad02cf1c8ab8e68cb52baa6a', '250ffcc789426a255f46d55537192b6', '312df229ef2675c4492d9da2668ec34c', '330dfd669062043f5e60ee22a227533', '341e24df542e0676d42b9650f19dd425', '379f0efc898d7a7e9fe74a48bbc553d7', '383bac847e38daa0e8dd9b2c07157c63', '388ffdd2dd3c0dc0780672cae8ec94c', '401e07ff01b1bfe64d85e9aeb23ad69a', '421b11a26931160a492d9da2668ec34c', '450b4a6b6afa6280d5a910bf1c23bf18', '462ef3bb97a88a753efde60c5bd5967e', '470ee88d9cabc6028be5ba60d0b948a6', '499c4b519c708ae84cd08aa7c510fb85', '507a5070cde81fd867936ca58e67cec6', '510a6ee3bc0a0669c8101121c12c3f9', '561d50862a2cb4aabc19762eaa7ba40f', '587ebb2aa71acfe644dd3aaee16d3f4c', '589cd6a1f4367fd834b707eb14b2a4a5', '590d04438aeffbb58f447453fccbd9d3', '640f61579181aef13ad3591a780fa12b', '659fdef8de4769079c251b4a7838829', '751b12c6e22386a49a195befe1c2db74', '774b94fd956a6601194565c2f3f896dc', '807f08096308af5e28c0cecb7de2397a', '815f436a40c28da51f56aa11cd5e0c3e', '902a46f5caf6b1f7e38c7f0de498b8e5', '910bbf8701a4846c2993aa294808121b', '975ea4be01c7488611bc8e8361bc5303', '986e49bd8314d7424addf6a5f8726274', '3427d2276fb713e5864c004309dc898d', '3896caf2057eed81589929e81650825e', '4460e0885d82b7b5d9987a7cd00eb98d', '4623efb456cc6070820018801b237b3d', '4841eb9b6b15d037dfd20491defa4a70', '5283a98b5c693e64ebefe6b1d594ad2e', '6534f04a1c349a3c8c6540fe6bc16d6f', '6678f63c9b584a549d9e5580ae9f8738', '6831b0aaec44c01f5d0b6dbf2c4992eb', '8748c7f2041761401fc5750ce9890422', '11740d372308f12185047f9f654ddc2e', '19861e56a952fe97b8230112437913fd', '24465c99afde3b9782f320c67d1f2e15', '33990ef5ffde80fa83bc207d8a5912e3', '37607ea19e352af4fffc97a61124b1a9', '39911f927331db1c8687ff9b0b4e4ac', '40747ff0ea92f134c409eaceb0331214', '46363a10e242999c69d8d5b35c38db0d', '46557f689f4cf5dd2acd2bb6205825cb', '51704a889ebcadb72cee4bf5d0f11ffe', '52621a97e5036dabba18ade30e563d37', '57263b7b778ac3a87c076cdc1de5871a', '58409b308683d908ca2bec46a3b47519', '58479a7b7c157865e68f66efebc71317', '76710f51915396e0327289c00b6dc9ca', '96929c12a4a6b15a492d9da2668ec34c', '224775a0c7f8fec65780b8d333445e3e', '276932e7bf229d37c8ad93869b52d215', '387600bd541f7b502d7ee37bd88bacc0', '439418b35f600f4bb10dc0fca58d0b2c', '458637d073cb6e9dc8687ff9b0b4e4ac', '513686d6d63a1d8e577b5d737869717e', '614347b0e864cdc7492d9da2668ec34c', '738395f54b301d80b1f5d603f931c1aa', '935093c683edbf2087946594df4e196c', '987122b945dcf1fef713f414dfd3c16', '2499541ace317cbb8cb5d9909aeb1309', '2608335ef22bcee1c97114870bb573', '2882587cd2fc021c168776226d349d71', '3075380de1f28ced3d290591da43d03e', '4706461c517a755d4e6ea835fbc6facc', '5042005e178d164481d0f12b8bf5c990', '7929676e756dcd41577b5d737869717e', '43897195d7f893d759c257be4c612509', '489391452ccb38e4c8687ff9b0b4e4ac', '568050037ad958625ea33df4b8846880', '624339070f8a749260c69156bdab9446', '892381333dc86d0196a8a62cbb17df9', '5923697231e1bc86af38a3b0483df58', '896303208667c3e9713decb1a0563b12', 'a3c4bd89e7570dcc6204511a506beabf', 'a3c1460f50bd470dd064b65ccd89778c', 'a3e990cd563b7a57ea411f5a15742da6', 'a3e985835e89c8b1498f847f9f26fae3', 'a5a5d118118eb760fee31d33418a3c16', 'a7aafe44f64dd04dfc6afbc561d74a4e', 'a64bc6079afa241f762ec387b772e9e1', 'a92a13b4968d7780472349060cd3c236', 'a128eda00983dd01fb7d9615be5ab4b0', 'a339a318c8a40fd4eebb133aa4be217e', 'a578b0027e7d9ec7b2ca3ea77e53abe', 'a631fb1b054697dcfd8aaf214a1df4be', 'a5898fefb1733333a82b0d8d157287f5', 'a9053230c8e02442c8687ff9b0b4e4ac', 'a9422478728741108e21d20fbef78328', 'aa05a0db9ab12c5e3e7d886a315f2350', 'aa5879b3f20ea90c36b0f2a1430e993a', 'acef669b784a4fec1b17743c18fb63dc', 'adfab5a4d020094f98cfd47a860803c5', 'af0831d39fd8f794492d9da2668ec34c', 'b3fd987b330d0d2acda56795a6fbde1f', 'b8fe0dd70bae3a9dba7a8c9ac8add14', 'b29f5c7c73908424685a4d2d2b5da82a', 'b33a3b1627ad61eb8ca4809dcf42fe1', 'b41aaea5754adae0444b41d6d7f557fa', 'b81be96660f38a8c7543d7dbfecfffbe', 'b89cbb45476e94a5e65235d5580cc3e0', 'b455c3037b013d85492d9da2668ec34c', 'b541d3afd111d86f265d8abf3d2f9c3', 'b0849a1c01da76eeffa41024c63fa9db', 'b987a2ca54c6ddecb74697ced5978572', 'b6843e186082096ebf80985a99195eb8', 'b59735c72cf9af0c17b431cae0dd70ed', 'b68180bfde45f5a7e450a02d8032dbb0', 'b80122c3a0543a7b7eaeab1f0c9120b7', 'ba5911cd07d1eebb31475fd256bf1b7a', 'bb660ca2ed9e0155f51763a096bdff1e', 'bb878313da4b02e3dbbc9440457e303e', 'bb90094030f369e4305a3b2fd9173d6f', 'bc6ca247b19398b89c7dd675565aedce', 'bbe63bca5688ac6b7236a8143b10600f', 'bcced2f12f206a927eaeab1f0c9120b7', 'bd6fcae754203396e34dfbaf829eddf7', 'bd0fab2e72b445bd1e722bceee6e83aa', 'bdaaebf065b112da492d9da2668ec34c', 'bdc58ca6d411823cae786e8787e8886a', 'be0c5a0e91c99e804e1a714ee619465a', 'be9ced795316066f38620af15fa7e604', 'be9d5105e48ae27e713decb1a0563b12', 'be1546c576c7c189edd7ee0476a331f6', 'be7897b96b04a5d12aca5d156344f8d3', 'bfa242a09f35dcaa38fba5f798d10731', 'c1b312919af633f8f51f77a6d7299806', 'c1c0b3d35cdef5e2f51fa0238791f5dc', 'c04c13649c4cbc5ebed0f345f50b6a5', 'c5ee6b77f9f84adeed52100e321c9f3e', 'c07c96f19ea1d431d4f67da0de6a1054', 'c7bbb1dfcac15f43e33c1bced7bc2a7', 'c9d68e1e5309ac25ac57e7d566628472', 'c50c2887ba2b5535c32fd7e88552aa9', 'c605d4742f68a00ece45069973024eb1', 'c0720c8d54745ea8c8687ff9b0b4e4ac', 'c747e6ceb1a6faaa3074f48b99186254', 'c976cb3eac6a89d9a0aa42b42238537d', 'c993ac723dbaf479a4929a32e32d23f3', 'c8265e04c94bcb5a1346e336f65f96f6', 'c67949ee2c241e8e9b9f2eb77f5e247e', 'c92721a95fe44b018039b09dacd0f1a7', 'ca01fd0de2534323c594a0e804f37c1a', 'ca764efc4e76a8b48d69616377752ecc', 'cbaca6a6edfa2d512b520984c067934c', 'cbcc5cd1aec9f3413aa677469bbdd68c', 'cc25ba35b3f6e8d3d064b65ccd89778c', 'cc30a723aeba69a139e0f39f5249b0ba', 'ccd5e24c9b96febd5208aab875b932bc', 'cd5ad4afabaed0d3e762624dc3c8fa2a', 'cd5cc7a5e50478e82b520984c067934c', 'cdfd278e8b1c11bfc36d58d0f13497a0', 'ce50c6235cf3ad8855afe589a9f09982', 'cfd2b4a2c8398220731d11154716a6b8', 'd2f844904a5cf31db93d537020ed867c', 'd6f2d44c693d2e857062f2d72cde5c95', 'd283b1073ceae03f35836c728d324152', 'd323e6d19dc58526f2c0effc06a15c11', 'd792c65f8b1710709b9f2eb77f5e247e', 'd2992fd5e6715bad3bbf93f83cbaf271', 'd3562f992aa405b214b1fd95dbca05', 'd6579d7c278cb9a47f093285855bdfd7', 'd8592dc28b9f0afa387379bbd607d69e', 'd29971cef754cc91cd8c5d1ba690a2c3', 'd38129a3301d31350b1fc43ca5e85e', 'de9e40c346ad2779f8c3d2002c77ddb', 'df23ca11080bb439676c272956dad3c2', 'df8440d8678f3a91c8687ff9b0b4e4ac', 'dfca4eae14e0bb541f6399f392c887ea', 'e1ca70ecc629eb46ca17c984d94453b4', 'e2ced471afce616454bfa32aa0766acb', 'e5ea39e2b4c28ea2a8df8437731d97b4', 'e08add655395ce8df51f77a6d7299806', 'e31d71ed32273fede42ac999db581f5e', 'e79b6815925c07acb0957d845ac33749', 'e0311d577cff9d85f6ea7461c0aa61c3', 'e19214cabca496a3f7b54e04c7238d7', 'ec91b1b29b6bccd37eaeab1f0c9120b7', 'ee5ee3f6759aabacf2f43e6f841bd32b', 'eeebe3fe14ee4d3aebefe6b1d594ad2e', 'efa83c67ce47bfca304edcf7c4314468', 'f2ef238e2e4615e92cf8e80367b40c9a', 'f3f0238ae8d9ba8dc8687ff9b0b4e4ac', 'f4b6bf9253918b52944d8f8e13d63fde', 'f4e24cf8c5d0c5c31dbb0393636b3531', 'f6cb298705842600bd259faf16bdac27', 'f51ab8433184dfd2c8687ff9b0b4e4ac', 'f52c7425a47ff55f79c8c47a54f30eb', 'f68ecc9ec512915f36d8dd30a594b2af', 'f645f79d47e2df0f7dca29e186afcbcf', 'f4268a28d2a837a1167c009da6daa010', 'f6810de4042cc5ce57bd4bc6eae9b341', 'f609378d4f5c755c9df2fbafa78e52d9', 'f6096649f87c37f1af7c7ad2549a1b15', 'fa263643fb3f45c7492d9da2668ec34c', 'faeb171add09f98cc8687ff9b0b4e4ac', 'fb8b45151900e6e016a0c57b9ceb6d01', 'fc0e86a20a9606493cd6abbbeca4e2ad', 'fd2c754ca43457107d910d51d878f708', 'fd726084724adeea492d9da2668ec34c', 'fdac1f9c0b030841c8687ff9b0b4e4ac', 'fdd21f7f2ca9f0bcbdcbca499b446e89', 'fdfedb5bb8cd35374233148ffd345970', 'ff6e6b7b7c7f1c0f15f78c98555ab41', 'ff8efd10f5e6c5c7c6c0380e62f2644', 'ffa6c1a509986a5676708f8dec712a28', 'fffda9f09223a21118ff2740a556cc3']
        print(len(all_white_ids))
        print(len(self.image_filenames))
        self.image_filenames = [image_filename for image_filename in self.image_filenames if os.path.basename(image_filename).split('_')[0] not in all_white_ids]
        print(len(self.image_filenames))
        
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
        image = Image.fromarray(np.uint8(np_image))

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