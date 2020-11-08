import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import lmdb
from tqdm import tqdm
from PIL import Image

from dataset import GeometryDataset, get_part_names, get_central_part_name
from VQVAE2Levels import VQVAE

import GeoVAE
import torchvision.models as models
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs', 'id_t', 'id_b', 'central_vggs'])

def extract_latents_patch(lmdb_env, loader, model, geo_model, vgg_model, args):
    index = 0
    transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.CenterCrop((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for i, (geo_input, origin_geo_input, logrmax, logrmin, smax, smin, filenames) in enumerate(pbar):
            geo_input = geo_input.to(device).float().contiguous()
            geo_zs, geo_outputs, _, _ = geo_model(geo_input)
            geo_zs = geo_zs.detach().cpu().numpy()

            for j in range(geo_input.shape[0]):
                filename = filenames[j]
                head_tail = os.path.split(filename)
                head = head_tail[0]
                basename = os.path.basename(filename)
                basename_without_ext = basename.split('.')[0]
                fid = basename_without_ext.split('_')[0]
                
                id_ts = []
                id_bs = []
                central_vggs = []
                for k in range(6):
                    # read image
                    img_filename = os.path.join(head, basename_without_ext+'_patch'+str(k)+'.png')
                    central_img_filename = os.path.join(head, fid+'_'+args.central_part_name+'_patch'+str(k)+'.png')
                    # print(img_filename)
                    if os.path.exists(img_filename):
                        img = Image.open(img_filename)
                        np_image = np.array(img)
                        np_image.setflags(write=1)
                        # np_image[:, :, 3] = np_image[:, :, 3]/255
                        # np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
                        # np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
                        # np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
                        # np_image[:, :, 3] = np_image[:, :, 3]*255
                        img = Image.fromarray(np.uint8(np_image[:, :, :3]))
                        img = transform(img)
                        img = img.to(device)
                        img.unsqueeze_(0)
                    else:
                        img = torch.zeros((1, 4, args.height, args.width), device=device)
                        print('warning: {} not exists'.format(img_filename))
                        continue

                    if os.path.exists(central_img_filename):
                        central_img = Image.open(central_img_filename)
                        np_central_image = np.array(central_img)
                        np_central_image.setflags(write=1)
                        # np_central_image[:, :, 3] = np_central_image[:, :, 3]/255
                        # np_central_image[:, :, 0] = np.multiply(np_central_image[:, :, 0], npcentral__image[:, :, 3])
                        # np_central_image[:, :, 1] = np.multiply(np_central_image[:, :, 1], npcentral__image[:, :, 3])
                        # np_central_image[:, :, 2] = np.multiply(np_central_image[:, :, 2], npcentral__image[:, :, 3])
                        # np_central_image[:, :, 3] = np_central_image[:, :, 3]*255
                        central_img = Image.fromarray(np.uint8(np_central_image[:, :, :3]))
                        central_img = transform(central_img)
                        central_img = central_img.to(device)
                        central_img.unsqueeze_(0)
                    else:
                        img = torch.zeros((1, 4, args.height, args.width), device=device)
                        print('warning: {} not exists'.format(central_img_filename))
                        continue
                    
                    # get indices
                    quant_t, quant_b, diff, id_t, id_b = model.encode(img)
                    # save
                    id_t = id_t.detach().cpu().numpy()
                    id_b = id_b.detach().cpu().numpy()
                    # append
                    id_ts.append(id_t)
                    id_bs.append(id_b)
                    # get vgg feature
                    central_vgg = vgg_model(central_img)
                    central_vgg = central_vgg.detach().cpu().numpy()
                    central_vggs.append(central_vgg)

                id_ts = np.concatenate(id_ts, axis=0)
                id_bs = np.concatenate(id_bs, axis=0)
                central_vggs = np.concatenate(central_vggs, axis=0)
                
                print('{} {} {} {}'.format(id_ts.shape, id_bs.shape, geo_zs[j:j+1, :].shape, central_vggs.shape))

                row = CodeRow(ID=filename, geo_zs=geo_zs[j:j+1, :], id_t=id_ts, id_b=id_bs, central_vggs=central_vggs)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=False)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--vqvae_ckpt_dir', type=str, required=True)

    parser.add_argument('--mat_dir', type=str, required=False)
    parser.add_argument('--ref_mesh_mat', type=str, required=False)
    parser.add_argument('--geo_hidden_dim', type=int, default=64)
    parser.add_argument('--geo_ckpt_dir', type=str, required=False)

    parser.add_argument('--category', type=str, required=False)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    part_names = get_part_names(args.category)
    central_part_name = get_central_part_name(args.category)
    part_names.remove(central_part_name)
    args.central_part_name = central_part_name

    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # torch pretrained vgg model 
    vgg16 = models.vgg16(pretrained=True).to(device)
    vgg16.eval()
    for part_name in part_names:
        part_save_path = os.path.join(args.save_path, part_name)
        if not os.path.exists(part_save_path):
            os.mkdir(part_save_path)

        # geometry dataset
        geo_dataset = GeometryDataset(args.mat_dir, part_name=part_name)
        geo_loader = DataLoader(geo_dataset, batch_size=16, shuffle=False, num_workers=1)

        # geo
        geo_model = GeoVAE.GeoVAE(geo_hidden_dim=args.geo_hidden_dim, ref_mesh_mat=args.ref_mesh_mat, device=device).to(device)
        geo_model = geo_model.float()
        
        geo_ckpt = os.path.join(args.geo_ckpt_dir, part_name, 'geovae_newest.pt')
        if not os.path.exists(geo_ckpt):
            print('{} not exists'.format(geo_ckpt))
            continue
        print('loading {}'.format(geo_ckpt))
        geo_model.load_state_dict(torch.load(geo_ckpt, map_location=torch.device(device)))
        geo_model.eval()

        # vqvae
        vqvae_ckpt = os.path.join(args.vqvae_ckpt_dir, part_name, 'vqvae_newest.pt')
        if not os.path.exists(vqvae_ckpt):
            print('{} not exists'.format(vqvae_ckpt))
            continue
        print('loading {}'.format(vqvae_ckpt))
        ckpt = torch.load(vqvae_ckpt, map_location=torch.device(device))
        vqvae_args = ckpt['args']
        model = VQVAE(in_channel=vqvae_args.in_channel,
            channel=vqvae_args.channel,
            n_res_block=vqvae_args.n_res_block,
            n_res_channel=vqvae_args.n_res_channel,
            embed_dim=vqvae_args.embed_dim,
            n_embed=vqvae_args.n_embed,
            decay=vqvae_args.decay,
            stride=vqvae_args.stride,).to(torch.device(device))
        model = model.float()
        model.load_state_dict(ckpt['model'])
        model.eval()

        map_size = 100 * 1024 * 1024 * 1024

        env = lmdb.open(part_save_path, map_size=map_size)
        args.device = device
        extract_latents_patch(env, geo_loader, model, geo_model, vgg16, args)
