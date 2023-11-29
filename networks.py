import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from torch import nn

from config import n

from config import T

class PositionEmbeddings(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.dim = dim  # dimensionality of pos embeddings. MUST be EVEN!
        self.n = n  # n param in pos encoding , 10000

        half_dim = self.dim // 2
        i = torch.arange(0, half_dim)
        denom = self.n ** (2 * i / self.dim)
        self.emb = torch.zeros((T, self.dim)).to(
            'cuda')  # move to gpu since t will be on gpu. not sure if its faster to instead copy t to cpu?
        for k in range(T):
            self.emb[k, 0::2] = torch.sin(k / denom)
            self.emb[k, 1::2] = torch.cos(k / denom)

        print(len(denom), len(self.emb[k, 0::2]), len(self.emb[k, 1::2]))

    def forward(self, t):
        # t: batch of timesteps shape=(batch_size,)
        # returns: embeddings shape=(batch_size, dim)
        #         device = t.device

        return self.emb[t]


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_dense = nn.Linear(time_emb_dim, out_ch)  # shape=(batchsize, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # shape=(batchsize, out_ch, H, W)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))  # shape=(batchsize, out_ch, H, W)
        # Time embedding
        time_emb = self.relu(self.time_dense(t))  # shape=(batchsize, out_ch)
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]  # shape=(batchsize, out_ch, 1, 1)
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            PositionEmbeddings(time_emb_dim, n),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        #             print(f"downs x {x.shape}")
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        #             print(f"ups x {x.shape}")
        return self.output(x)


