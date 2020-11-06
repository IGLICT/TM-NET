import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from LatentsDatasetWithGeoVGG2levels import LatentsDatasetWithGeoVGG2levels
from pixelsnail import PixelSNAIL, PixelSNAILTop
from scheduler import CycleScheduler
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs', 'id_t', 'id_b', 'central_vggs'])

def max_min_normalize(AA, min_v, max_v):
    (B, H, W) = AA.shape
    AA = AA.view(B, -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(B, H, W)
    AA = AA * (max_v - min_v) + min_v
    return AA

def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (ID, geo_zs, top, bottom, central_vggs) in enumerate(loader):
        model.zero_grad()
        
        # geo_zs = max_min_normalize(geo_zs, 0, 255)
        # geo_zs = geo_zs.round()
        # geo_zs = geo_zs.repeat(1, 128*6, 1)
        # geo_zs = geo_zs.unsqueeze_(1)
        # geo_zs = F.interpolate(geo_zs, size=(48, 8))
        # geo_zs = geo_zs.squeeze(1)
        # geo_zs = geo_zs.to(torch.long)

        # geo_zs = max_min_normalize(geo_zs, 0, 255)
        # geo_zs = geo_zs.round()
        # geo_zs = geo_zs.squeeze(1)
        # temp_geo_zs = torch.zeros((geo_zs.shape[0], 48*8))
        # temp_geo_zs[:, :geo_zs.shape[1]] = geo_zs
        # temp_geo_zs = temp_geo_zs.reshape(geo_zs.shape[0], 48, 8)
        # geo_zs = temp_geo_zs
        # geo_zs = geo_zs.to(torch.long)

        geo_zs = geo_zs.unsqueeze(1)
        central_vggs = central_vggs.reshape((central_vggs.shape[0], -1)).contiguous()
        central_vggs = central_vggs.unsqueeze(1)
        central_vggs = central_vggs.unsqueeze(1)
        condition_input = torch.cat((geo_zs, central_vggs), 3)
        condition_input = condition_input.to(device)

        # top = top.squeeze(1)
        # bottom = bottom.squeeze(1)
        top = top.reshape(top.shape[0], top.shape[1]*top.shape[2], top.shape[3])
        bottom = bottom.reshape(bottom.shape[0], bottom.shape[1]*bottom.shape[2], bottom.shape[3])
        # print('{} {} {}'.format(top.shape, bottom.shape, geo_zs.shape))
        if args.hier == 'top':
            top = top.to(device).contiguous()
            geo_zs = geo_zs.to(device).contiguous()
            target = top
            out, _ = model(top, condition=condition_input)
        elif args.hier == 'bottom':
            top = top.to(device).contiguous()
            bottom = bottom.to(device).contiguous()
            target = bottom
            out, _ = model(bottom, condition=top)
        
        # loss = criterion(out, target)
        # loss.backward()
        cross_entropy_loss, _ = model.backward(out, target)

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}/{args.epoch}; loss: {cross_entropy_loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--hier', type=str, default='toptop')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O1')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_ckpt_dir', type=str, default='./condition_ckpt')
    parser.add_argument('--part_name', type=str, default='body')
    args = parser.parse_args()
    # torch.backends.cudnn.enabled = False

    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device
    print(args)

    dataset = LatentsDatasetWithGeoVGG2levels(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    old_args = None
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        print('{} loaded'.format(args.ckpt))
        old_args = ckpt['args']
    if old_args is None:
        old_args = args

    if args.hier == 'top':
        model = PixelSNAILTop(
            [96, 16],
            256,
            # args.channel,
            128,
            5,
            4,
            old_args.n_res_block,
            # old_args.n_res_channel,
            128,
            dropout=old_args.dropout,
            n_out_res_block=old_args.n_out_res_block,
            n_cond_res_block=old_args.n_cond_res_block,
            cond_res_channel=old_args.n_res_channel,
            n_condition_dim=64+1000*6,
            n_condition_class=2000
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [192, 32],
            256,
            old_args.channel,
            # 128,
            5,
            4,
            old_args.n_res_block,
            # old_args.n_res_channel,
            128,
            attention=False,
            dropout=old_args.dropout,
            n_cond_res_block=old_args.n_cond_res_block,
            cond_res_channel=old_args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None and args.use_amp == True:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    # model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    if not os.path.exists(args.save_ckpt_dir):
        os.mkdir(args.save_ckpt_dir)
    sub_ckpt_dir = os.path.join(args.save_ckpt_dir, args.part_name)
    if not os.path.exists(sub_ckpt_dir):
        os.mkdir(sub_ckpt_dir)

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, scheduler, device)
        torch.save(
            # {'model': model.module.state_dict(), 'args': args},
            {'model': model.state_dict(), 'args': args},
            f'{sub_ckpt_dir}/pixelsnail_{args.hier}.pt',
        )
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
