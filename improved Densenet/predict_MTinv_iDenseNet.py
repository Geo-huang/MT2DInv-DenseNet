import torch
import torchvision
from PIL import Image
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import  pandas  as pd
from sklearn import preprocessing
import scipy.signal
import random
import torch.nn.functional as F
import torchvision.ops as ops

# 设置全局字体为新罗马字体
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"


i=100 #测试样本


image_path = './test/{}.dat'.format(i)
image = np.loadtxt(image_path, delimiter=',')
label = image[:2176]
image = image[2176:]
data = image.reshape(16, -1)
data = image.reshape(16, 33,-1)
# print(data)
# 视电阻率归一化
rho1_norm = data[:, :, 0] - np.average(data[:, :, 0])
# print(rho1_norm)
# print(np.max(abs(rho1_norm)))
rho1_norm = rho1_norm / np.max(abs(rho1_norm))
rho1_norm = torch.Tensor(rho1_norm)
# print(rho1_norm)
rho2_norm = data[:, :, 2] - np.average(data[:, :, 2])
rho2_norm = rho2_norm / np.max(abs(rho2_norm))
rho2_norm = torch.Tensor(rho2_norm)
# print(rho2_norm)
# 相位归一化
phase1_norm = data[:, :, 1] - np.average(data[:, :, 1])
phase1_norm = phase1_norm / np.max(abs(phase1_norm))
phase1_norm = torch.Tensor(phase1_norm)
# print(phase1_norm)
phase2_norm = data[:, :, 3] - np.average(data[:, :, 3])
phase2_norm = phase2_norm / np.max(abs(phase2_norm))
phase2_norm = torch.Tensor(phase2_norm)
# print(phase2_norm)
x = torch.cat((rho1_norm.unsqueeze(-1), phase1_norm.unsqueeze(-1), rho2_norm.unsqueeze(-1), phase2_norm.unsqueeze(-1)),
              dim=-1)  # 将归一化后的视电阻率和相位拼接起来
# print(x)

x = x.reshape(1,16, 33, -1)
image = x

# image = torch.Tensor(x)



target_data = np.asarray(label).reshape(34,64)
# true_resistivity = target_data
# print(target_data)
target_data  =np.log10(target_data )
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.imshow((target_data), cmap='turbo_r', vmin=-0, vmax=4, extent=[-16, 16, 17, 0])
cbar = plt.colorbar(shrink=0.7)
cbar.ax.set_ylabel('Rho     log10(ohm-m)', fontsize=18)
cbar.ax.yaxis.set_label_coords(-1.5, 0.5)
cbar.ax.tick_params(labelsize=18)
plt.xlabel('X/Km', fontsize=22)
plt.ylabel('Depth/Km', fontsize=22)
plt.xticks(np.arange(-16, 17, 4), fontsize=22)
plt.yticks(np.arange(0, 17, 5), fontsize=22)



class ICBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ICBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):

        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        x = x * channel_att * spatial_att  #
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2  # 保持特征图大小不变的填充

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
    def forward(self, x):
        return self.layer(x)


class DenseNetWithICBAM(nn.Module):
    def __init__(self, num_blocks, growth_rate, num_classes):
        super(DenseNetWithICBAM, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3)
        self.dense_blocks = nn.ModuleList()
        in_channels = 64

        for i, num_layers in enumerate(num_blocks):
            self.dense_blocks.append(DenseBlock(in_channels, growth_rate, num_layers))
            in_channels += num_layers * growth_rate

            if i == 3:  # Add ICBAM after the second DenseBlock
                self.dense_blocks.append(ICBAM(in_channels))

            if i < len(num_blocks) - 1:
                out_channels = in_channels // 2
                self.dense_blocks.append(TransitionLayer(in_channels, out_channels))
                in_channels = out_channels

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.dense_blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
model = DenseNetWithICBAM(num_blocks=[6, 12, 24, 16], growth_rate=32, num_classes=2176)
model.load_state_dict(torch.load('MTinv_iDenseNet.pth'))

model.eval()
with torch.no_grad():

    output = np.asarray(model(image)).reshape(34,64)

    # print(10**output)

    # 计算均方误差
    mse = np.mean((output - target_data) ** 2)
    print("Mean Squared Error:", mse)


    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.imshow((output), cmap='turbo_r', vmin=-0, vmax=4, extent=[-16, 16, 17, 0])
    cbar = plt.colorbar(shrink=0.7)
    cbar.ax.set_ylabel('Rho     log10(ohm-m)', fontsize=18)
    cbar.ax.yaxis.set_label_coords(-1.5, 0.5)
    cbar.ax.tick_params(labelsize=18)
    plt.xlabel('X/Km', fontsize=22)
    plt.ylabel('Depth/Km', fontsize=22)
    plt.xticks(np.arange(-16, 17, 4), fontsize=22)
    plt.yticks(np.arange(0, 17, 5), fontsize=22)
    plt.show()





