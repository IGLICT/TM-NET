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

class PixelSNAILOthersAgent(BaseAgent):
    def __init__(self, config):
        super(PixelSNAILOthersAgent, self).__init__(config)
        self.device = config[config['mode']]['device']
        self.hier = config['model']['name'].split('_')[1]
        print(config['train']['lr'])
        
    def build_net(self, config):
        net = get_network(config)
        print('-----Part Sequence architecture-----')
        print(net)
        if config['data']['parallel']:
            net = nn.DataParallel(net, device_ids=config['data']['parallel_devices'])
        # net = net.to(self.device)
        return net

    def set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        outputs = {}
        losses = {}
        geo_zs, top, bottom, central_vggs, filenames = data
        geo_zs = geo_zs.unsqueeze(1)
        # top = top.reshape(top.shape[0], top.shape[2], top.shape[1]*top.shape[3])
        # bottom = bottom.reshape(bottom.shape[0], bottom.shape[2], bottom.shape[1]*bottom.shape[3])
        top = top.reshape(top.shape[0], top.shape[1]*top.shape[2], top.shape[3])
        bottom = bottom.reshape(bottom.shape[0], bottom.shape[1]*bottom.shape[2], bottom.shape[3])
        central_vggs = central_vggs.reshape(central_vggs.shape[0], 1, 1, 6000)
        # print(top.shape)
        # print(bottom.shape)
        if self.hier == 'top':
            top = top.to(self.device).contiguous()
            geo_zs = geo_zs.to(self.device).contiguous()
            central_vggs = central_vggs.to(self.device).contiguous()
            # print(geo_zs.shape)
            # print(central_vggs.shape)
            concated_condition = torch.cat([central_vggs, geo_zs], -1)
            # print(concated_condition.shape)
            target = top
            out, _, latent_diff = self.net(top, condition=concated_condition)
        elif self.hier == 'bottom':
            top = top.to(self.device).contiguous()
            bottom = bottom.to(self.device).contiguous()
            target = bottom
            out, _, latent_diff = self.net(bottom, condition=top)
        # cross_entropy_loss
        CE_loss = self.criterion(out, target)
        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        print(accuracy)

        outputs['out'] = out
        losses['CE'] = CE_loss
        if latent_diff is not None:
            losses['latent'] = latent_diff
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
        recon_dir = os.path.join(self.model_dir, '{}_recon'.format(mode))
        if not os.path.exists(recon_dir):
            os.mkdir(recon_dir)
        pass
