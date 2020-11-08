import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.pool import edge_pool, TopKPooling
import GraphConvyj as yj

import scipy.io as sio
import numpy as np
import math

class PartFeatSampler(nn.Module):

    def __init__(self, in_size, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic
        middle_dim = 4096

        # self.linear = nn.Linear(in_size, middle_dim, bias = False)
        self.mlp2mu = nn.Linear(in_size, feature_size, bias = False)
        self.mlp2var = nn.Linear(in_size, feature_size, bias = False)
        # self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, x):
        # x = self.linear(x)
        mu = self.mlp2mu(x)

        if self.probabilistic:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu

class PartDeformEncoder(nn.Module):

    def __init__(self, num_point, feat_len, edge_index = None, probabilistic=True, bn = True):
        super(PartDeformEncoder, self).__init__()
        self.probabilistic = probabilistic
        self.bn = bn
        self.edge_index = edge_index

        self.conv1 = yj.GCNConv(9, 9, edge_index)
        self.conv2 = yj.GCNConv(9, 9, edge_index)
        self.conv3 = yj.GCNConv(9, 9, edge_index)
        # self.conv4 = yj.GCNConv(9, 9, edge_index)
        # self.conv5 = yj.GCNConv(9, 9, edge_index)
        # self.conv6 = yj.GCNConv(9, 9, edge_index)
        # self.conv7 = yj.GCNConv(9, 9, edge_index)
        # self.conv8 = yj.GCNConv(9, 9, edge_index)
        # self.conv9 = yj.GCNConv(9, 9, edge_index)
        # self.conv10 = yj.GCNConv(9, 9, edge_index)
        

        if self.bn:
            self.bn1 = torch.nn.InstanceNorm1d(9)
            self.bn2 = torch.nn.InstanceNorm1d(9)
            self.bn3 = torch.nn.InstanceNorm1d(9)
            # self.bn4 = torch.nn.InstanceNorm1d(9)
            # self.bn5 = torch.nn.InstanceNorm1d(9)
            # self.bn6 = torch.nn.InstanceNorm1d(9)
            # self.bn7 = torch.nn.InstanceNorm1d(9)
            # self.bn8 = torch.nn.InstanceNorm1d(9)
            # self.bn9 = torch.nn.InstanceNorm1d(9)
            # self.bn10 = torch.nn.InstanceNorm1d(9)
        self.mlp2mu = nn.Linear(num_point*9, feat_len)
        self.mlp2var = nn.Linear(num_point*9, feat_len)

        # self.sampler = PartFeatSampler(in_size = num_point*9, feature_size=feat_len, probabilistic=probabilistic)

    def forward(self, featurein):
        feature = featurein
        self.vertex_num = feature.shape[1]
        # print(self.vertex_num)
        if self.bn:
            net = nn.functional.leaky_relu(self.bn1(self.conv1(feature)), negative_slope=0)
            net = nn.functional.leaky_relu(self.bn2(self.conv2(net)), negative_slope=0)
            net = nn.functional.leaky_relu(self.bn3(self.conv3(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn4(self.conv4(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn5(self.conv5(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn6(self.conv6(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn7(self.conv7(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn8(self.conv8(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn9(self.conv9(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn10(self.conv10(net)), negative_slope=0)
        else:
            net = nn.functional.leaky_relu(self.conv1(feature), negative_slope=0)
            net = nn.functional.leaky_relu(self.conv2(net), negative_slope=0)
            net = nn.functional.leaky_relu(self.conv3(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv4(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv5(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv6(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv7(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv8(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv9(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv10(net), negative_slope=0)
            # net = torch.tanh(self.conv3(net, edge_index))
        net = net.contiguous().view(-1, self.vertex_num * 9)

        # net = self.sampler(net)
        mu = self.mlp2mu(net)
        logvar = self.mlp2var(net)

        return mu, logvar

class PartDeformDecoder(nn.Module):

    def __init__(self, feat_len, num_point, edge_index= None, bn = True):
        super(PartDeformDecoder, self).__init__()
        self.num_point = num_point
        self.bn = bn
        middle_dim = 2048

        # self.mlp2 = nn.Linear(feat_len, middle_dim, bias = False)
        self.mlp1 = nn.Linear(feat_len, self.num_point * 9, bias = False)
        self.conv1 = yj.GCNConv(9, 9, edge_index)
        self.conv2 = yj.GCNConv(9, 9, edge_index)
        self.conv3 = yj.GCNConv(9, 9, edge_index)
        # self.conv4 = yj.GCNConv(9, 9, edge_index)
        # self.conv5 = yj.GCNConv(9, 9, edge_index)
        # self.conv6 = yj.GCNConv(9, 9, edge_index)
        # self.conv7 = yj.GCNConv(9, 9, edge_index)
        # self.conv8 = yj.GCNConv(9, 9, edge_index)
        # self.conv9 = yj.GCNConv(9, 9, edge_index)
        # self.conv10 = yj.GCNConv(9, 9, edge_index)

        if bn:
            self.bn1 = torch.nn.InstanceNorm1d(9)
            self.bn2 = torch.nn.InstanceNorm1d(9)
            self.bn3 = torch.nn.InstanceNorm1d(9)
            # self.bn4 = torch.nn.InstanceNorm1d(9)
            # self.bn5 = torch.nn.InstanceNorm1d(9)
            # self.bn6 = torch.nn.InstanceNorm1d(9)
            # self.bn7 = torch.nn.InstanceNorm1d(9)
            # self.bn8 = torch.nn.InstanceNorm1d(9)
            # self.bn9 = torch.nn.InstanceNorm1d(9)
            # self.bn10 = torch.nn.InstanceNorm1d(9)

        self.L2Loss = nn.L1Loss(reduction = 'mean')

    def forward(self, net):
        # printprint(self.num_point)
        # net = self.mlp2(net)
        net = self.mlp1(net).view(-1, self.num_point, 9)
        if self.bn:
            net = nn.functional.leaky_relu(self.bn1(self.conv1(net)), negative_slope=0)
            net = nn.functional.leaky_relu(self.bn2(self.conv2(net)), negative_slope=0)
            net = nn.functional.leaky_relu(self.bn3(self.conv3(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn4(self.conv4(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn5(self.conv5(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn6(self.conv6(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn7(self.conv7(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn8(self.conv8(net)), negative_slope=0)
            # net = nn.functional.leaky_relu(self.bn9(self.conv9(net)), negative_slope=0)
        else:
            net = nn.functional.leaky_relu(self.conv1(net), negative_slope=0)
            net = nn.functional.leaky_relu(self.conv2(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv3(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv4(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv5(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv6(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv7(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv8(net), negative_slope=0)
            # net = nn.functional.leaky_relu(self.conv9(net), negative_slope=0)
        net = torch.tanh(self.conv3(net))

        return net

    def loss(self, pred, gt):
        avg_loss = self.L2Loss(pred, gt) * 100000

        return avg_loss

class GeoVAE(nn.Module):
    """docstring for GeoVAE"""
    def __init__(self, 
        geo_hidden_dim=128, 
        ref_mesh_mat='../guitar_with_mapping.mat',
        device='cpu'):
        super(GeoVAE, self).__init__()
        self.geo_hidden_dim = geo_hidden_dim
        self.ref_mesh_mat = ref_mesh_mat
        self.device = torch.device(device)

        ref_mesh_data = sio.loadmat(self.ref_mesh_mat)
        V = ref_mesh_data['V']
        F = ref_mesh_data['F']
        edge_index = ref_mesh_data['edge_index'].astype(np.int64).transpose()
        edge_index = torch.from_numpy(edge_index).to(self.device)

        self.num_point = V.shape[0]
        print(self.num_point)
        self.geo_encoder = PartDeformEncoder(self.num_point, self.geo_hidden_dim, edge_index, probabilistic=False, bn = False)
        self.geo_decoder = PartDeformDecoder(self.geo_hidden_dim, self.num_point, edge_index, bn = False)

    def encode(self, geo_input):
        mu, logvar = self.geo_encoder(geo_input)
        mu, logvar = mu.contiguous(), logvar.contiguous()
        return mu, logvar

    def decode(self, geo_z):
        geo_output = self.geo_decoder(geo_z).contiguous()
        return geo_output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, geo_input):
        mu, logvar = self.encode(geo_input)
        geo_z = self.reparameterize(mu, logvar)
        geo_output = self.decode(geo_z)

        return geo_z, geo_output, mu, logvar
    
class GeoVAEAllParts(nn.Module):
    """docstring for GeoVAE"""
    def __init__(self, 
        geo_hidden_dim=128, 
        part_num=3,
        ref_mesh_mat='../guitar_with_mapping.mat',
        device='cpu'
        ):
        super(GeoVAEAllParts, self).__init__()
        self.geo_hidden_dim = geo_hidden_dim
        self.ref_mesh_mat = ref_mesh_mat
        self.device = torch.device(device)
        self.part_num = part_num

        ref_mesh_data = sio.loadmat(self.ref_mesh_mat)
        V = ref_mesh_data['V']
        F = ref_mesh_data['F']
        edge_index = ref_mesh_data['edge_index'].astype(np.int64).transpose()
        edge_index = torch.from_numpy(edge_index).to(self.device)

        self.num_point = V.shape[0]
        print(self.num_point)
        # self.leaves =  nn.ModuleList([nn.Linear(ni, num_classes) for i in range(self.num_leaves)])
        # self.nodes =  nn.ModuleList([nn.Linear(ni, 1) for i in range(self.num_nodes)])

        self.geo_encoders = nn.ModuleList([PartDeformEncoder(self.num_point, self.geo_hidden_dim, edge_index, probabilistic=False, bn = True) for i in range(self.part_num)])
        self.geo_decoders = nn.ModuleList([PartDeformDecoder(self.geo_hidden_dim, self.num_point, edge_index, bn = True) for i in range(self.part_num)])
        # for i in range(part_num):
        #     geo_encoder = PartDeformEncoder(self.num_point, self.geo_hidden_dim, edge_index, probabilistic=False, bn = True)
        #     geo_decoder = PartDeformDecoder(self.geo_hidden_dim, self.num_point, edge_index, bn = True)
        #     self.geo_encoders.append(geo_encoder)
        #     self.geo_decoders.append(geo_decoder)

    def encode(self, geo_inputs):
        geo_zs = torch.zeros((geo_inputs.shape[0], geo_inputs.shape[1], self.geo_hidden_dim), dtype=torch.float).to(self.device)
        for i in range(self.part_num):
            geo_zs[:, i, :] = self.geo_encoders[i](geo_inputs[:, i, :, :]).contiguous()
        return geo_zs

    def decode(self, geo_zs):
        geo_outputs = torch.zeros((geo_zs.shape[0], geo_zs.shape[1], self.num_point, 9), dtype=torch.float).to(self.device)
        for i in range(self.part_num):
            geo_outputs[:, i, :, :] = self.geo_decoders[i](geo_zs[:, i, :]).contiguous()
        return geo_outputs

    def forward(self, geo_inputs):
        geo_zs = self.encode(geo_inputs)
        geo_outputs = self.decode(geo_zs)

        return geo_zs, geo_outputs