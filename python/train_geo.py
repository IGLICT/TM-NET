import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from scheduler import CycleScheduler

from PIL import Image
import torchvision.transforms.functional as F
import os
from dataset import GeometryDataset
import scipy.io as sio
import numpy as np
import GeoVAE

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    MSELoss = nn.MSELoss(reduction='mean')
    for i, (geo_input, origin_geo_input, logrmax, logrmin, smax, smin, fullname) in enumerate(loader):
        model.zero_grad()
        geo_input = geo_input.to(device).float().contiguous()
        geo_z, geo_output, mu, logvar = model(geo_input)
        # geo_output[geo_output != geo_output] = 0
        geo_recon_loss = MSELoss(geo_input, geo_output)*model.num_point*9
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/geo_output.shape[0]
        
        total_loss = geo_recon_loss + kl_loss*0.001
        total_loss.backward()
        if torch.isinf(geo_recon_loss):
            print(filename)
            
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1} '
                f'geo_recon: {geo_recon_loss.item():.1f} '
                f'kl: {kl_loss.item():.1f} '
                f'lr: {lr:.1f}'
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ref_mesh_mat', type=str, required=True)
    parser.add_argument('--mat_dir', type=str, required=True)
    parser.add_argument('--part_name', type=str, required=True)
    parser.add_argument('--geo_hidden_dim', type=int, default=64)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--is_test', type=int, default=0)
    parser.add_argument('--load_ckpt', action='store_true')
    
    args = parser.parse_args()
    print(args)

    # device = 'cpu'
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device
    print(device)

    dataset = GeometryDataset(args.mat_dir, args.part_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = GeoVAE.GeoVAE(geo_hidden_dim=args.geo_hidden_dim, ref_mesh_mat=args.ref_mesh_mat, device=device).to(device)
    model = model.float()
    
    # for p in model.parameters():
    #     print(p.name, p.dtype)
    checkpoint_dir = os.path.join(args.ckpt_dir, args.part_name)
    if args.load_ckpt and args.ckpt_dir is not None:
        checkpoint_dir = args.ckpt_dir
    if not os.path.exists(checkpoint_dir) and args.is_test == False:
        os.makedirs(checkpoint_dir)

    if args.is_test == 0:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = None
        if args.sched == 'cycle':
            scheduler = CycleScheduler(
                optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
            )

        # model.load_state_dict(torch.load(f'./checkpoint/vqvae_newest.pt'))
        if args.ckpt_dir is not None and args.load_ckpt == True:
            model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.part_name, 'geovae_newest.pt'), map_location=device))
        
        for i in range(args.epoch):
            train(i, loader, model, optimizer, scheduler, device)
            if (i+1) % 1 == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'geovae_newest.pt'))
    elif args.is_test == 1:
        dataset = GeometryDataset(args.mat_dir, args.part_name)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        if args.ckpt_dir is not None:
            model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'geovae_newest.pt'), map_location=args.device))
        recon_dir = os.path.join(args.ckpt_dir, 'recon_dir')
        if not os.path.exists(recon_dir):
            os.mkdir(recon_dir)

        for i, (geo_input, origin_geo_input, logrmax, logrmin, smax, smin, id_name) in enumerate(loader):
            id_name = id_name[0]
            id_name = os.path.basename(id_name)
            id_name = id_name.split('.')[0]

            geo_input = geo_input.to(device).float()
            geo_z, geo_output, mu, logvar = model(geo_input)

            origin_geo_input = torch.Tensor.cpu(origin_geo_input).detach().numpy()
            geo_input = torch.Tensor.cpu(geo_input).detach().numpy()
            geo_output = torch.Tensor.cpu(geo_output).detach().numpy()
            def normalize_back(logrmax, logrmin, smax, smin, geo_output):
                logr = geo_output[:, :, :3]
                s = geo_output[:, :, 3:9]
                resultmax = 0.95
                resultmin = -0.95

                s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
                logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin
                geo_output = np.concatenate((logr, s), axis = 2)
                return geo_output
            origin_geo_output = normalize_back(logrmax, logrmin, smax, smin, geo_output)
            print('{} {} {}'.format(id_name, np.linalg.norm(origin_geo_input-origin_geo_output), np.linalg.norm(geo_input-geo_output)))
            sio.savemat(os.path.join(recon_dir, id_name+'.mat'), {'geo_output': origin_geo_output}, do_compression=False)
    elif args.is_test == 2:
        pass
    elif args.is_test == 3:
        if args.ckpt_dir is not None:
            model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'geovae_newest.pt'), map_location=device))
        recon_dir = os.path.join(args.ckpt_dir, 'random_dir')
        if not os.path.exists(recon_dir):
            os.mkdir(recon_dir)

        for i in range(100):
            geo_z = torch.from_numpy(np.random.rand(1, args.geo_hidden_dim))
            geo_z = geo_z.to(device).float()
            geo_output = model.decode(geo_z)
            sio.savemat(os.path.join(recon_dir, str(i)+'.mat'), {'geo_output': torch.Tensor.cpu(geo_output).detach().numpy()}, do_compression=False)