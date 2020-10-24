import math

import torch
from torch import nn
from torch.nn import functional as F

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        # print(flatten.shape)
        # torch.Size([1024, 64])

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # print(dist.shape)
        # torch.Size([1024, 512])

        _, embed_ind = (-dist).max(1)
        # print(embed_ind.shape)
        # torch.Size([1024])

        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # print(embed_onehot.shape)
        # torch.Size([1024, 512])

        embed_ind = embed_ind.view(*input.shape[:-1])
        # print(embed_ind.shape)
        # torch.Size([1, 32, 32])

        quantize = self.embed_code(embed_ind)
        # print(quantize.shape)
        # torch.Size([1, 32, 32, 64])

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            # print(self.cluster_size.shape)
            # torch.Size([512])

            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            # print(embed_sum.shape)
            # torch.Size([64, 512])


            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            # print(self.embed_avg.shape)
            # torch.Size([64, 512])

            n = self.cluster_size.sum()
            # print(n.shape)
            # torch.Size([])

            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            # print(cluster_size.shape)
            # torch.Size([512])

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            # print(embed_normalized.shape)
            # torch.Size([64, 512])
            
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]
        elif stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 16:
            blocks = [
                nn.Conv2d(in_channel, channel // 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )
        elif stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 4, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 4, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        elif stride == 16:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, channel // 4, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 4, channel // 8, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 8, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=4,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=32,
        decay=0.99,
        stride=4,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=stride,
        )
        # bottom_stride = 2
        # image_size = 256
        # self.quant_t_encoder = FeatureMapEncoder(image_size//bottom_stride//4, 64, 3, 128)
        # self.quant_t_decoder = FeatureMapDecoder(image_size//bottom_stride//4, 1024, 3, 128)
        # self.quant_b_encoder = FeatureMapEncoder(image_size//bottom_stride//2, 64, 3, 256)
        # self.quant_b_decoder = FeatureMapDecoder(image_size//bottom_stride//2, 1024, 3, 256)

    def forward(self, input):
        quant_t, quant_b, diff, id_t, id_b = self.encode(input)
        # print(id_t.shape)
        # print(id_b.shape)
        dec = self.decode(quant_t, quant_b)
        # dec1 = self.decode(quant_t1, quant_b1)

        # return dec, diff, quant_t, quant_b, dec1, diff1, quant_t1, quant_b1
        return dec, diff, quant_t, quant_b

    def encode(self, input):
        # print(input.shape)
        # torch.Size([1, 3, 256, 256])

        enc_b = self.enc_b(input)
        # print(enc_b.shape)
        # torch.Size([1, 128, 64, 64])

        enc_t = self.enc_t(enc_b)
        # print(enc_t.shape)
        # torch.Size([1, 128, 32, 32])

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        # print(quant_t.shape)
        # torch.Size([1, 32, 32, 64])

        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        # print(quant_t.shape)
        # torch.Size([1, 64, 32, 32])

        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        # print(dec_t.shape)
        # torch.Size([1, 64, 64, 64])

        enc_b = torch.cat([dec_t, enc_b], 1)
        # print(enc_b.shape)
        # torch.Size([1, 192, 64, 64])

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        # print(quant_b.shape)
        # torch.Size([1, 64, 64, 64])

        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        # print(quant_b.shape)
        # torch.Size([1, 64, 64, 64])

        diff_b = diff_b.unsqueeze(0)

        # feature map recon AE
        # quant_t_z = self.quant_t_encoder(quant_t)
        # recon_quant_t = self.quant_t_decoder(quant_t_z)
        # quant_b_z = self.quant_b_encoder(quant_b)
        # recon_quant_b = self.quant_b_decoder(quant_b_z)
        
        # quant_t1, diff_t1, id_t1 = self.quantize_t(recon_quant_t.permute(0, 2, 3, 1))
        # quant_t1 = quant_t1.permute(0, 3, 1, 2)
        # quant_b1, diff_b1, id_b1 = self.quantize_b(recon_quant_b.permute(0, 2, 3, 1))
        # quant_b1 = quant_b1.permute(0, 3, 1, 2)

        # return quant_t, quant_b, diff_t + diff_b, id_t, id_b, quant_t1, quant_b1, diff_t1 + diff_b1, id_t1, id_b1
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
