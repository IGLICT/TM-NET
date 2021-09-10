import torch
from torch import nn
from torch.nn import functional as F
import scipy.io as sio
import numpy as np
import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

class SPVAEEncoder(nn.Module):
    """docstring for SPVAEEncoder"""
    def __init__(self, feat_len):
        super(SPVAEEncoder, self).__init__()
        self.feat_len = feat_len

        self.mlp1 = torch.nn.Linear(feat_len, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.leakyrelu1 = torch.nn.LeakyReLU()

        self.mlp2 = torch.nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.leakyrelu2 = torch.nn.LeakyReLU()

        self.mlp3 = torch.nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.leakyrelu3 = torch.nn.LeakyReLU()

        self.mlp_mu = torch.nn.Linear(256, 128)
        self.sigmoid_mu = torch.nn.Sigmoid()

        self.mlp_logvar = torch.nn.Linear(256, 128)

    def forward(self, featurein):
        featureout = self.leakyrelu1(self.bn1(self.mlp1(featurein)))
        featureout = self.leakyrelu2(self.bn2(self.mlp2(featureout)))
        featureout = self.mlp3(featureout)
        mu = self.sigmoid_mu(self.mlp_mu(featureout))
        logvar = self.mlp_logvar(featureout)
        return mu, logvar

class SPVAEDecoder(nn.Module):
    """docstring for SPVAEDecoder"""
    def __init__(self, feat_len):
        super(SPVAEDecoder, self).__init__()
        self.feat_len = feat_len

        self.mlp1 = torch.nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.leakyrelu1 = torch.nn.LeakyReLU()
        
        self.mlp2 = torch.nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.leakyrelu2 = torch.nn.LeakyReLU()

        self.mlp3 = torch.nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(num_features=1024)
        self.leakyrelu3 = torch.nn.LeakyReLU()

        self.mlp4 = torch.nn.Linear(1024, self.feat_len)
        self.tanh = torch.nn.Tanh()

    def forward(self, featurein):
        featureout = self.leakyrelu1(self.bn1(self.mlp1(featurein)))
        featureout = self.leakyrelu2(self.bn2(self.mlp2(featureout)))
        featureout = self.leakyrelu3(self.bn3(self.mlp3(featureout)))
        featureout = self.tanh(self.mlp4(featureout))
        return featureout
        

class SPVAE(nn.Module):
    """docstring for SPVAE"""
    def __init__(self, 
        geo_hidden_dim=64,
        part_num=7,
        device='cpu'):
        super(SPVAE, self).__init__()
        self.geo_hidden_dim = geo_hidden_dim
        self.part_num = part_num
        self.feat_len = self.part_num*(self.part_num*2+9+self.geo_hidden_dim)
        self.encoder = SPVAEEncoder(feat_len=self.feat_len)
        self.decoder = SPVAEDecoder(feat_len=self.feat_len)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        return  z, self.decoder(z), mu, log_var
    
    def loss_function(self,
                      *args,
                      kld_weight=0.001) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}