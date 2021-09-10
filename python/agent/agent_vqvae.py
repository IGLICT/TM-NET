import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from networks import get_network
from torchvision import utils

from agent.base import BaseAgent
from util.visualization import merge_patches

class VQVAEAgent(BaseAgent):
    def __init__(self, config):
        super(VQVAEAgent, self).__init__(config)
        self.device = config[config['mode']]['device']
        # print(config['train']['lr'])
        self.in_channel = config['model']['in_channel']
        self.latent_loss_weight = config['model']['alpha']

    def build_net(self, config):
        net = get_network(config)
        print('-----Part Sequence architecture-----')
        print(net)
        if config['data']['parallel']:
            net = nn.DataParallel(net)
        print(self.device)
        net = net.to(self.device)
        return net

    def set_loss_function(self):
        self.criterion = nn.L1Loss(reduction='sum')

    def get_seam_loss(self, recon_batches):
        if recon_batches.shape[0] % 6 != 0:
            print('batch size shoule be set as a multiply of 6.')
            return 0
        model_num = recon_batches.shape[0]/6
        loss = 0

        for i in range(int(model_num)):
            patch0 = recon_batches[6*i, :, :, :]
            patch1 = recon_batches[6*i+1, :, :, :]
            patch2 = recon_batches[6*i+2, :, :, :]
            patch3 = recon_batches[6*i+3, :, :, :]
            patch4 = recon_batches[6*i+4, :, :, :]
            patch5 = recon_batches[6*i+5, :, :, :]

            loss += (
                self.criterion(patch0[:, :, 0], patch1[:, 0, :]) + \
                self.criterion(patch0[:, 255, :], patch2[:, 0, :]) + \
                self.criterion(patch0[:, :, 255], torch.flip(patch3[:, 0, :], [1])) + \
                self.criterion(patch0[:, 0, :], torch.flip(patch4[:, 0, :], [1])) + \
                self.criterion(patch1[:, :, 255], patch2[:, :, 0]) + \
                self.criterion(patch1[:, :, 0], patch4[:, :, 255]) + \
                self.criterion(patch1[:, 255, :], torch.flip(patch5[:, :, 0], [1])) + \
                self.criterion(patch2[:, :, 255], patch3[:, :, 0]) + \
                self.criterion(patch2[:, 255, :], patch5[:, 0, :]) + \
                self.criterion(patch3[:, :, 255], patch4[:, :, 0]) + \
                self.criterion(patch3[:, 255, :], patch5[:, :, 255]) + \
                self.criterion(patch4[:, 255, :], torch.flip(patch5[:, 255, :], [1])) \
                )/model_num
        return loss

    def forward(self, data):
        outputs = {}
        losses = {}
        latent_loss_weight = self.latent_loss_weight
        
        img = rearrange(data[0], 'B P C H W -> (B P) C H W')
        filenames = data[1]

        img = img.to(self.device).contiguous()
        dec, latent_loss, quant_t, quant_b = self.net(img)
        
        # print('{} {} '.format(quant_t.shape, quant_b.shape))
        # texture
        recon_loss = self.criterion(dec, img)/img.shape[0]
        latent_loss = latent_loss.mean() * 64 * 16 * 16
        # seam_loss
        seam_loss = self.get_seam_loss(dec)

        outputs['dec'] = dec
        losses['recon'] = recon_loss
        losses['latent'] = latent_loss * latent_loss_weight
        return outputs, losses

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)
        self.record_losses(losses, 'train')

        return outputs, losses

    def visualize_batch(self, data, mode, outputs=None):
        if mode == 'train':
            return
        imgs = data[0]
        filenames = data[1]
        # flat
        flat_filenames = []
        for i in range(len(filenames[0])):
            for j in range(len(filenames)):
                flat_filenames.append(filenames[j][i])
        filenames = flat_filenames

        recon_dir = os.path.join(self.model_dir, '{}_recon'.format(mode))
        if not os.path.exists(recon_dir):
            os.mkdir(recon_dir)
        
        dec = outputs['dec']
        # if dec.shape[1] == 4:
        #     dec[:, 3, :, :] = ((dec[:, 3, :, :] > 0).float()-0.5)*2
        dec = dec.clamp(-1, 1)
        for i in range(dec.shape[0]):
            filename = filenames[i]
            utils.save_image(
                dec[i, :, :, :],
                # imgs[i, :, :, :],
                os.path.join(recon_dir, filename+'.png'),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
        merge_patches(recon_dir, channel=self.in_channel)