import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from VQVAE2Levels import VQVAE
from scheduler import CycleScheduler

from PIL import Image
import torchvision.transforms.functional as F
import os
from dataset import PatchImageDataset
import scipy.io as sio
import numpy as np
import GeoVAE
import glob
import cv2

def merge_patches(patch_image_dir):
    patch0_files = glob.glob(os.path.join(patch_image_dir, '*patch0.png*'))
    merged_dir = os.path.join(patch_image_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)

    for patch0_file in patch0_files:
        head_tail = os.path.split(patch0_file)
        head = head_tail[0]
        tail = head_tail[1]

        patch0 = cv2.imread(patch0_file, cv2.IMREAD_UNCHANGED)
        # get filename
        patch1_file = os.path.join(head, tail.replace('patch0', 'patch1'))
        patch2_file = os.path.join(head, tail.replace('patch0', 'patch2'))
        patch3_file = os.path.join(head, tail.replace('patch0', 'patch3'))
        patch4_file = os.path.join(head, tail.replace('patch0', 'patch4'))
        patch5_file = os.path.join(head, tail.replace('patch0', 'patch5'))
        # read
        patch1 = cv2.imread(patch1_file, cv2.IMREAD_UNCHANGED)
        patch2 = cv2.imread(patch2_file, cv2.IMREAD_UNCHANGED)
        patch3 = cv2.imread(patch3_file, cv2.IMREAD_UNCHANGED)
        patch4 = cv2.imread(patch4_file, cv2.IMREAD_UNCHANGED)
        patch5 = cv2.imread(patch5_file, cv2.IMREAD_UNCHANGED)
        # indices
        H_begin = [256, 0, 256, 512, 768, 256]
        W_begin = [0, 256, 256, 256, 256, 512]
        patches = []
        patches.append(patch0)
        patches.append(patch1)
        patches.append(patch2)
        patches.append(patch3)
        patches.append(patch4)
        patches.append(patch5)
        # merge
        merged_image = np.zeros((768, 1024, 3), np.uint8)
        for i in range(6):
            try:
                merged_image[W_begin[i]:W_begin[i]+256, H_begin[i]:H_begin[i]+256, :] = patches[i]
            except Exception:
                continue
        # save
        out_name = os.path.join(merged_dir, tail.replace('patch0', ''))
        cv2.imwrite(out_name, merged_image)

def get_seam_loss(recon_batches):
    if recon_batches.shape[0] % 6 != 0:
        print('batch size shoule be set as a multiply of 6.')
        return 0
    model_num = recon_batches.shape[0]/6
    loss = 0
    L1Loss = nn.L1Loss(reduction='sum')

    for i in range(int(model_num)):
        patch0 = recon_batches[6*i, :, :, :]
        patch1 = recon_batches[6*i+1, :, :, :]
        patch2 = recon_batches[6*i+2, :, :, :]
        patch3 = recon_batches[6*i+3, :, :, :]
        patch4 = recon_batches[6*i+4, :, :, :]
        patch5 = recon_batches[6*i+5, :, :, :]

        loss += (
               L1Loss(patch0[:, :, 0], patch1[:, 0, :]) + \
               L1Loss(patch0[:, 255, :], patch2[:, 0, :]) + \
               L1Loss(patch0[:, :, 255], torch.flip(patch3[:, 0, :], [1])) + \
               L1Loss(patch0[:, 0, :], torch.flip(patch4[:, 0, :], [1])) + \
               L1Loss(patch1[:, :, 255], patch2[:, :, 0]) + \
               L1Loss(patch1[:, :, 0], patch4[:, :, 255]) + \
               L1Loss(patch1[:, 255, :], torch.flip(patch5[:, :, 0], [1])) + \
               L1Loss(patch2[:, :, 255], patch3[:, :, 0]) + \
               L1Loss(patch2[:, 255, :], patch5[:, 0, :]) + \
               L1Loss(patch3[:, :, 255], patch4[:, :, 0]) + \
               L1Loss(patch3[:, 255, :], patch5[:, :, 255]) + \
               L1Loss(patch4[:, 255, :], torch.flip(patch5[:, 255, :], [1])) \
            )/model_num
    return loss


def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    MSELoss = nn.MSELoss(reduction='sum')
    L1Loss = nn.L1Loss(reduction='sum')

    latent_loss_weight = 0.25
    sample_size = 1

    mse_sum = 0
    mse_n = 0

    for i, (img, filename) in enumerate(loader):
        model.zero_grad()

        img = img.to(device).contiguous()
        dec, latent_loss, quant_t, quant_b = model(img)
        # print('{} {} '.format(quant_t.shape, quant_b.shape))
        # texture
        recon_loss = L1Loss(dec, img)/img.shape[0]
        latent_loss = latent_loss.mean() * 64 * 16 * 16
        # seam_loss
        seam_loss = get_seam_loss(dec)

        loss = recon_loss + latent_loss_weight * latent_loss + seam_loss
        loss.backward(retain_graph=True)

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item()
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1} '
                f'mse: {recon_loss.item():.1f} '
                f'latent: {latent_loss.item():.1f} '
                f'seam: {seam_loss.item():.1f} '
                f'lr: {lr:.4f}'
            )
        )


if __name__ == '__main__':
    # texture
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_test', type=int, default=0, help='0: training 1: reconstruction, 2: interpolation, 3: random 4: auto-texture')
    # training
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--part_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--load_ckpt', action='store_true')
    # vqvae model
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--n_res_block', type=int, default=2)
    parser.add_argument('--n_res_channel', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_embed', type=int, default=256)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--num_point', type=int, default=9602)
    parser.add_argument('--geo_hidden_dim', type=int, default = 128)
    parser.add_argument('--ref_mesh_mat', type=str, default='../data/all/car_std.mat')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    # geometry
    parser.add_argument('--mat_dir', type=str, default=64)
    parser.add_argument('--geo_ckpt_dir', type=str)

    # condition
    parser.add_argument('--noise_Z_dim', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--mapping_ckpt_dir', type=str)

    args = parser.parse_args()

    print(args)

    # device = 'cpu'
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device
    print(device)
    

    transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.CenterCrop((args.height, args.width)),
            # data augmentation
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if args.is_test == 0:
        model = VQVAE(
            in_channel=args.in_channel,
            channel=args.channel,
            n_res_block=args.n_res_block,
            n_res_channel=args.n_res_channel,
            embed_dim=args.embed_dim,
            n_embed=args.n_embed,
            decay=args.decay,
            stride=args.stride,
            ).to(torch.device(args.device))
        model = model.float()
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        sub_ckpt_dir = os.path.join(args.ckpt_dir, args.part_name)
        if not os.path.exists(sub_ckpt_dir):
            os.mkdir(sub_ckpt_dir)

        dataset = PatchImageDataset(args.image_dir, transform=transform, part_name=args.part_name)
        # dataset = PatchImageDataset(args.image_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = None
        if args.sched == 'cycle':
            scheduler = CycleScheduler(
                optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
            )
        if args.ckpt_dir is not None and args.load_ckpt is True:
            ckpt = torch.load(os.path.join(args.ckpt_dir, 'vqvae_newest.pt'), map_location=device)
            model.load_state_dict(ckpt['model'])
            print('load checkpoint successfully')
        
        for i in range(args.epoch):
            train(i, loader, model, optimizer, scheduler, device)
            if (i+1) % 1 == 0:
                torch.save({'model': model.state_dict(), 'args': args}, os.path.join(sub_ckpt_dir, 'vqvae_newest.pt'))
    else:
        model = VQVAE(
            in_channel=args.in_channel,
            channel=args.channel,
            n_res_block=args.n_res_block,
            n_res_channel=args.n_res_channel,
            embed_dim=args.embed_dim,
            n_embed=args.n_embed,
            decay=args.decay,
            stride=args.stride,
            ).to(torch.device(args.device))

        model = model.float()

        print('loading {}'.format(os.path.join(args.ckpt_dir, 'vqvae_newest.pt')))
        ckpt = torch.load(os.path.join(args.ckpt_dir, 'vqvae_newest.pt'), map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        # if args.image_dir.endswith('train'):
        #     recon_dir = os.path.join(args.ckpt_dir, 'train_recon1')
        #     auto_texture_dir = os.path.join(args.ckpt_dir, 'train_auto_texture')
        # elif args.image_dir.endswith('test'):
        #     recon_dir = os.path.join(args.ckpt_dir, 'test_recon1')
        #     auto_texture_dir = os.path.join(args.ckpt_dir, 'test_auto_texture')
        # else:
        #     recon_dir = os.path.join(args.ckpt_dir, 'test_recon1')
        #     auto_texture_dir = os.path.join(args.ckpt_dir, 'test_auto_texture')
        #     print('image_dir does not end with \'train or \'test')
             
        # if not os.path.exists(recon_dir):
        #     os.mkdir(recon_dir)
        # if not os.path.exists(auto_texture_dir):
        #     os.mkdir(auto_texture_dir)
        transform = transforms.Compose(
            [
                transforms.Resize((args.height, args.width)),
                transforms.CenterCrop((args.height, args.width)),
                # data augmentation
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        # recon_latent reconstruction
        if args.is_test == 1:
            if args.image_dir.endswith('train'):
                recon_dir = os.path.join(args.ckpt_dir, 'train_recon')
            elif args.image_dir.endswith('test'):
                recon_dir = os.path.join(args.ckpt_dir, 'test_recon')
            if not os.path.exists(recon_dir):
                os.mkdir(recon_dir)

            # test_dataset = ImageFileDataset(args.image_dir, transform=transform)
            test_dataset = PatchImageDataset(args.image_dir, transform=transform, part_name=args.part_name)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
            for i, (img, filename) in enumerate(test_loader):
                filename = filename[0]

                # if '2dbc73ad4ce7950163e148e250c0340d' not in filename:
                #     continue
                img = img.to(device)
                
                quant_t, quant_b, diff, id_t, id_b = model.encode(img)
                dec = model.decode(quant_t, quant_b)
                # dec[:, 3, :, :] = ((dec[:, 3, :, :] > 0).float()-0.5)*2

                utils.save_image(
                    # torch.cat([img, dec, recon_dec], 0),
                    dec,
                    # os.path.join(recon_dir, filename+'_0.png'),
                    os.path.join(recon_dir, filename+'.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            merge_patches(recon_dir)
            
        # interpolation
        elif args.is_test == 2:
            interpolation_dir = os.path.join(args.ckpt_dir, 'interpolation')
            if not os.path.exists(interpolation_dir):
                os.mkdir(interpolation_dir)
            id1 = 'a276d9eee2bb79f2691c0d594e383a87'
            id2 = 'aface94c7aeb373865ffdcf45b6f330c'
            
            # part_names = ['part1', 'part2']
            part_names = ['part1', 'part2', 'part3']
            # part_names = ['part1', 'part2']
            # part_names = ['body', 'left_front_wheel', 'right_front_wheel', 'left_back_wheel', 'right_back_wheel','left_mirror','right_mirror']
            # part_names = ['left_leg1', 'left_leg2', 'left_leg3', 'left_leg4', 'right_leg1', 'right_leg2', 'right_leg3', 'right_leg4', 'surface']
            # part_names = ['back', 'hand_1', 'hand_2', 'leg_ver_1', 'leg_ver_2', 'leg_ver_3', 'leg_ver_4', 'seat']

            interpolation_num = 11
            for part_name in part_names:
                filename1 = os.path.join(args.image_dir, id1, id1+'_'+part_name+'.png')
                filename2 = os.path.join(args.image_dir, id2, id2+'_'+part_name+'.png')
                if not os.path.exists(filename1) or not os.path.exists(filename2):
                    continue
                image1 = Image.open(filename1)
                image2 = Image.open(filename2)
                np_image = np.array(image1)
                np_image.setflags(write=1)
                np_image[:, :, 3] = np_image[:, :, 3]/255
                np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
                np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
                np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
                np_image[:, :, 3] = np_image[:, :, 3]*255
                image1 = Image.fromarray(np.uint8(np_image))
                x1 = transform(image1)
                # x1 = F.to_tensor(image1)
                x1.unsqueeze_(0)
                # x1 = x1*2 - 1
                x1 = x1.to(device)

                np_image = np.array(image2)
                np_image.setflags(write=1)
                np_image[:, :, 3] = np_image[:, :, 3]/255
                np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
                np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
                np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
                np_image[:, :, 3] = np_image[:, :, 3]*255
                image2 = Image.fromarray(np.uint8(np_image))
                x2 = transform(image2)
                # x2 = F.to_tensor(image2)
                x2.unsqueeze_(0)
                # x2 = x2*2 - 1
                x2 = x2.to(device)

                quant_tt1, quant_t1, quant_b1, diff1, id_tt1, id_t1, id_b1, z_tt1, z_t1, z_b1 = model.encode(x1)
                quant_tt2, quant_t2, quant_b2, diff2, id_tt2, id_t2, id_b2, z_tt2, z_t2, z_b2 = model.encode(x2)

                # toptop
                quant_tt1 = quant_tt1.reshape([-1, 64*32*32])
                quant_tt2 = quant_tt2.reshape([-1, 64*32*32])
                # top
                quant_t1 = quant_t1.reshape([-1, 64*64*64])
                quant_t2 = quant_t2.reshape([-1, 64*64*64])
                # geo_z_quant1 = geo_z_quant1.reshape([-1, 6*6*64])
                quant_b1 = quant_b1.reshape([-1, 64*128*128])
                quant_b2 = quant_b2.reshape([-1, 64*128*128])
                # geo_z_quant2 = geo_z_quant2.reshape([-1, 6*6*64])


                inter_weights = torch.linspace(0, 1, steps=interpolation_num)
                inter_weights = inter_weights.reshape([-1, 1])
                inter_weights = inter_weights.to(device)

                inter_quant_tts = torch.lerp(quant_tt1, quant_tt2, inter_weights)
                inter_quant_ts = torch.lerp(quant_t1, quant_t2, inter_weights)
                inter_quant_bs = torch.lerp(quant_b1, quant_b2, inter_weights)
                # inter_geo_z_quants = torch.lerp(geo_z_quant1, geo_z_quant2, inter_weights)

                inter_quant_tts = inter_quant_tts.reshape([-1, 64, 32, 32])
                inter_quant_ts = inter_quant_ts.reshape([-1, 64, 64, 64])
                inter_quant_bs = inter_quant_bs.reshape([-1, 64, 128, 128])
                dec = model.decode(inter_quant_tts, inter_quant_ts, inter_quant_bs)

                dec_1 = model.decode(quant_tt1.reshape([-1, 64, 32, 32]), quant_t1.reshape([-1, 64, 64, 64]), quant_b1.reshape([-1, 64, 128, 128]))
                dec_2 = model.decode(quant_tt2.reshape([-1, 64, 32, 32]), quant_t2.reshape([-1, 64, 64, 64]), quant_b2.reshape([-1, 64, 128, 128]))
                utils.save_image(
                    dec_1,
                    # f'interpolation/{part_name}_{str(i)}.png',
                    os.path.join(interpolation_dir, id1+'_'+part_name+'.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    dec_2,
                    # f'interpolation/{part_name}_{str(i)}.png',
                    os.path.join(interpolation_dir, id2+'_'+part_name+'.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                for i in range(interpolation_num):
                    utils.save_image(
                        dec[i, :, :, :],
                        # f'interpolation/{part_name}_{str(i)}.png',
                        os.path.join(interpolation_dir, part_name+str(i)+'.png'),
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                
        #         sio.savemat(os.path.join('interpolation', part_name+'_'+str(i)+'.mat'), {'geo_output': torch.Tensor.cpu(geo_output[i:i+1, :, :]).detach().numpy()})
        # generation
        elif args.is_test == 3:
            if args.image_dir.endswith('train'):
                auto_texture_dir = os.path.join(args.mapping_ckpt_dir, 'train_auto_texture')
            elif args.image_dir.endswith('test'):
                auto_texture_dir = os.path.join(args.mapping_ckpt_dir, 'test_auto_texture')
            else:
                print('image_dir does not end with \'train or \'test')
            random_dir = os.path.join(args.ckpt_dir, 'random')
            if not os.path.exists(random_dir):
                os.mkdir(random_dir)
            generation_num = 200
            embed_dim = 64

            toptop_latent_dim = 256
            top_latent_dim = 512
            bottom_latent_dim = 2048
            z_tt_latent = torch.randn(generation_num, toptop_latent_dim)
            z_t_latent = torch.randn(generation_num, top_latent_dim)
            z_b_latent = torch.randn(generation_num, bottom_latent_dim)

            quant_tt = model.z_tt_decoder(z_tt_latent)
            quant_t = model.z_t_decoder(z_t_latent)
            quant_b = model.z_b_decoder(z_b_latent)

            lookup_quant_tt, _, _ = model.quantize_tt(quant_tt.permute(0, 2, 3, 1))
            lookup_quant_t, _, _ = model.quantize_t(quant_t.permute(0, 2, 3, 1))
            lookup_quant_b, _, _ = model.quantize_b(quant_b.permute(0, 2, 3, 1))

            lookup_quant_tt = lookup_quant_tt.permute(0, 3, 1, 2)
            lookup_quant_t = lookup_quant_t.permute(0, 3, 1, 2)
            lookup_quant_b = lookup_quant_b.permute(0, 3, 1, 2)

            dec = model.decode(lookup_quant_tt, lookup_quant_t, lookup_quant_b)
            for i in range(generation_num):
                utils.save_image(
                    dec[i, :, :, :],
                    # f'random/{str(i)}.png',
                    os.path.join(random_dir, str(i)+'.png'),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
        # auto texture
        elif args.is_test == 4:
            geo_dataset = GeometryDatasetAllPartsNewTensor(args.mat_dir, args.part_name, args.ref_mesh_mat)
            geo_loader = DataLoader(geo_dataset, batch_size=1, shuffle=True, num_workers=4)

            part_num = len(geo_dataset.part_names)
            geo_model = GeoVAE.GeoVAEAllParts(geo_hidden_dim=args.geo_hidden_dim, part_num=part_num, ref_mesh_mat=args.ref_mesh_mat, device=device).to(device)
            geo_model = geo_model.float()
            print('loading {}'.format(args.geo_ckpt_dir))
            geo_model.load_state_dict(torch.load(args.geo_ckpt_dir, map_location=torch.device(device)))
            geo_model.eval()

            print('loading {}'.format(args.mapping_ckpt_dir))
            mapping_ckpt = torch.load(args.mapping_ckpt_dir, map_location=device)

            # mapping_toptop = Mapping.BicycleGANAllParts(args.geo_hidden_dim, 256, args.noise_Z_dim, part_num, device=device)
            # mapping_toptop.to(device)
            # mapping_top = Mapping.BicycleGANAllParts(args.geo_hidden_dim, 512, args.noise_Z_dim, part_num, device=device)
            # mapping_top.to(device)
            # mapping_bottom = Mapping.BicycleGANAllParts(args.geo_hidden_dim, 2048, args.noise_Z_dim, part_num, device=device)
            # mapping_bottom.to(device)

            num_layers = 6
            mapping_toptop = Mapping.BicycleGAN(args.geo_hidden_dim, 32*32, args.noise_Z_dim, num_layers=num_layers, device=device)
            mapping_toptop.to(device)
            mapping_top = Mapping.BicycleGAN(args.geo_hidden_dim, 64*64, args.noise_Z_dim, num_layers=num_layers, device=device)
            mapping_top.to(device)
            mapping_bottom = Mapping.BicycleGAN(args.geo_hidden_dim, 128*128, args.noise_Z_dim, num_layers=num_layers, device=device)
            mapping_bottom.to(device)

            mapping_toptop.load_state_dict(mapping_ckpt['toptop'])
            mapping_top.load_state_dict(mapping_ckpt['top'])
            mapping_bottom.load_state_dict(mapping_ckpt['bottom'])

            mapping_toptop.eval()
            mapping_top.eval()
            mapping_bottom.eval()

            # args = mapping_ckpt['args']
            MSELoss = nn.MSELoss(reduction='sum')

            for i, (geo_inputs, filename) in enumerate(geo_loader):
                filename = filename[0]

                geo_inputs = geo_inputs.to(device).float()
                geo_zs, geo_outputs = geo_model(geo_inputs)
                geo_zs = geo_zs.squeeze(axis=0)
                # geo_zs[geo_zs!=geo_zs]=0

                imgs = torch.zeros((part_num, 4, args.size, args.size), device=device)

                flag = 1
                l = 0
                for part_name in geo_dataset.part_names:
                    img_filename = os.path.join(args.image_dir, filename+'_'+part_name+'.png')
                    if os.path.exists(img_filename):
                        img = Image.open(img_filename)
                        np_image = np.array(img)
                        np_image.setflags(write=1)
                        np_image[:, :, 3] = np_image[:, :, 3]/255
                        np_image[:, :, 0] = np.multiply(np_image[:, :, 0], np_image[:, :, 3])
                        np_image[:, :, 1] = np.multiply(np_image[:, :, 1], np_image[:, :, 3])
                        np_image[:, :, 2] = np.multiply(np_image[:, :, 2], np_image[:, :, 3])
                        np_image[:, :, 3] = np_image[:, :, 3]*255
                        img = Image.fromarray(np.uint8(np_image))
                        img = transform(img)
                        # img.unsqueeze_(0)
                        img = img.to(device)
                    else:
                        img = torch.zeros((4, args.size, args.size), device=device)
                        flag = 0
                    imgs[l, :, :, :] = img
                    l = l + 1

                    if flag == 0:
                        break
                if flag == 0:
                    continue

                quant_tt, quant_t, quant_b, diff, id_tt, id_t, id_b, z_tt, z_t, z_b, z_tt_latent, z_t_latent, z_b_latent, recon_z_tt, recon_z_t, recon_z_b, recon_quant_tt, recon_quant_t, recon_quant_b = model.encode(imgs)

                mapping_toptop.set_input(geo_zs, id_tt.reshape(-1, 32*32).to(torch.float))
                mapping_top.set_input(geo_zs, id_t.reshape(-1, 64*64).to(torch.float))
                mapping_bottom.set_input(geo_zs, id_b.reshape(-1, 128*128).to(torch.float))

                # mapping_toptop.set_input(geo_zs, z_tt_latent)
                # mapping_top.set_input(geo_zs, z_t_latent)
                # mapping_bottom.set_input(geo_zs, z_b_latent)

                for j in range(1):
                    # toptop_fake_B = mapping_toptop.test(encode=False)
                    # top_fake_B = mapping_top.test(encode=False)
                    # bottom_fake_B = mapping_bottom.test(encode=False)

                    _, toptop_fake_B, toptop_real_B = mapping_toptop.test(encode=True)
                    _, top_fake_B, top_real_B = mapping_top.test(encode=True)
                    _, bottom_fake_B, bottom_real_B = mapping_bottom.test(encode=True)

                    toptop_fake_B = toptop_fake_B.round().to(torch.int64).reshape(-1, 32, 32).contiguous()
                    top_fake_B = top_fake_B.round().to(torch.int64).reshape(-1, 64, 64).contiguous()
                    bottom_fake_B = bottom_fake_B.round().to(torch.int64).reshape(-1, 128, 128).contiguous()

                    toptop_fake_B[toptop_fake_B>=256]=255
                    top_fake_B[top_fake_B>=256]=255
                    bottom_fake_B[bottom_fake_B>=256]=255
                    toptop_fake_B[toptop_fake_B<0]=0
                    top_fake_B[top_fake_B<0]=0
                    bottom_fake_B[bottom_fake_B<0]=0

                    recon_quant_tt = model.quantize_tt.embed_code(toptop_fake_B)
                    recon_quant_t = model.quantize_t.embed_code(top_fake_B)
                    recon_quant_b = model.quantize_b.embed_code(id_b)
                    recon_quant_tt = recon_quant_tt.permute(0, 3, 1, 2).contiguous()
                    recon_quant_t = recon_quant_t.permute(0, 3, 1, 2).contiguous()
                    recon_quant_b = recon_quant_b.permute(0, 3, 1, 2).contiguous()
                    
                    # predicted_recon_z_tt = model.z_tt_decoder(toptop_real_B)
                    # predicted_recon_z_t = model.z_t_decoder(top_real_B)
                    # predicted_recon_z_b = model.z_b_decoder(bottom_real_B)

                    # predicted_recon_z_tt = predicted_recon_z_tt.permute(0, 2, 3, 1)
                    # predicted_recon_z_t = predicted_recon_z_t.permute(0, 2, 3, 1)
                    # predicted_recon_z_b = predicted_recon_z_b.permute(0, 2, 3, 1)

                    # recon_quant_tt, _, _ = model.quantize_tt(predicted_recon_z_tt)
                    # recon_quant_tt = recon_quant_tt[:, :, :, 0:model.embed_dim].contiguous()
                    # recon_quant_tt = recon_quant_tt.permute(0, 3, 1, 2).contiguous()

                    # recon_quant_t, _, _ = model.quantize_t(predicted_recon_z_t)
                    # recon_quant_t = recon_quant_t[:, :, :, 0:model.embed_dim].contiguous()
                    # recon_quant_t = recon_quant_t.permute(0, 3, 1, 2).contiguous()
                    
                    # recon_quant_b, _, _ = model.quantize_b(predicted_recon_z_b)
                    # recon_quant_b = recon_quant_b[:, :, :, 0:model.embed_dim].contiguous()
                    # recon_quant_b = recon_quant_b.permute(0, 3, 1, 2).contiguous()


                    # dec = model.decode(quant_tt, quant_t, quant_b)
                    dec = model.decode(recon_quant_tt, recon_quant_t, recon_quant_b)
                    dec[:, 3, :, :] = ((dec[:, 3, :, :] > 0).float()-0.5)*2
                    # print(dec.min(), dec.max())
                    # recon_featuremap_loss =  MSELoss(z_tt.flatten(start_dim=1), predicted_recon_z_tt.flatten(start_dim=1)) + MSELoss(z_t.flatten(start_dim=1), predicted_recon_z_t.flatten(start_dim=1)) + MSELoss(predicted_recon_z_b.flatten(start_dim=1), z_b.flatten(start_dim=1))
                    # print(recon_featuremap_loss)
                    for k in range(part_num):
                        utils.save_image(
                                # torch.cat([img, dec], 0),
                                dec[k, :, :, :],
                                f'{auto_texture_dir}/{filename}_{geo_dataset.part_names[k]}_{j}.png',
                                nrow=1,
                                normalize=True,
                                range=(-1, 1),
                            )
            # sio.savemat(os.path.join(train_auto_texture, filename+'.mat'), {'geo_output': torch.Tensor.cpu(geo_output).detach().numpy()}, do_compression=False)

        # condition
        # for i, (img, geo_input, filename) in enumerate(test_loader):
        #     img = img.to(device)
        #     geo_input = geo_input.to(device).float()
        #     condition_input = img[:, :, 127:128, 127:128].permute(0, 2, 3, 1)
        #     quant_t, quant_b, diff, id_t, id_b, geo_z, z_t, z_b, recon_z_t, recon_z_b, predicted_z_t_latent, predicted_z_b_latent, predicted_recon_z_t, predicted_recon_z_b = model.encode(img, geo_input, condition_input)

        #     # predicted_recon_z_t = predicted_recon_z_t.reshape(-1, 16, 16, 64)
        #     predicted_recon_z_t = predicted_recon_z_t.reshape(-1, 32, 32, 64)
        #     z_t_condition = torch.cat((predicted_recon_z_t, condition_input.repeat(1, predicted_recon_z_t.shape[1], predicted_recon_z_t.shape[2], 1)), 3).contiguous()
        #     quant_t, diff_t, id_t = model.quantize_t(z_t_condition)
        #     quant_t = quant_t[:, :, :, 0:64].contiguous()
        #     quant_t = quant_t.permute(0, 3, 1, 2).contiguous()

        #     # predicted_recon_z_b = predicted_recon_z_b.reshape(-1, 32, 32, 64)
        #     predicted_recon_z_b = predicted_recon_z_b.reshape(-1, 64, 64, 64)
        #     z_b_condition = torch.cat((predicted_recon_z_b, condition_input.repeat(1, predicted_recon_z_b.shape[1], predicted_recon_z_b.shape[2], 1)), 3).contiguous()
        #     quant_b, diff_b, id_b = model.quantize_b(z_b_condition)
        #     quant_b = quant_b[:, :, :, 0:64].contiguous()
        #     quant_b = quant_b.permute(0, 3, 1, 2).contiguous()


        #     dec, geo_output = model.decode(quant_t, quant_b, geo_z)
        #     utils.save_image(
        #             torch.cat([img, dec], 0),
        #             # out,
        #             f'train_auto_texture/{filename}.png',
        #             nrow=1,
        #             normalize=True,
        #             range=(-1, 1),
        #         )
        #     sio.savemat(os.path.join('./train_auto_texture', filename[0]+'.mat'), {'geo_output': torch.Tensor.cpu(geo_output).detach().numpy()}, do_compression=False)

        
        # reconstruction
        # for i, (img, geo_input, filename) in enumerate(test_loader):
        #     filename = filename[0]
            
        #     img = img.to(device)
        #     geo_input = geo_input.to(device).float()
        #     condition_input = img[:, :, 127:128, 127:128].permute(0, 2, 3, 1)
            
        #     dec, diff, geo_output, z_t, z_b, recon_z_t, recon_z_b, recon_quant_t, recon_quant_b, recon_dec = model(img, geo_input, condition_input)

        #     utils.save_image(
        #         # torch.cat([img, out], 0),
        #         dec,
        #         # os.path.join(train_recon_dir, filename+'.png'),
        #         os.path.join(test_recon_dir, filename+'.png'),
        #         nrow=1,
        #         normalize=True,
        #         range=(-1, 1),
        #     )
            # print(geo_output)
        #     sio.savemat(os.path.join(train_recon_dir, filename+'.mat'), {'geo_output': torch.Tensor.cpu(geo_output).detach().numpy()}, do_compression=False)

        
