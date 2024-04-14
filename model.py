import os
import numpy as np
import os.path as osp
import cv2
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F
import math
from transformer import *
from thop import profile
"""
For further compression speed and accelerated memory access, 
the another BN layer as well as the weight fusion part are not included in this code.
"""
# ----------------------------------------------------------------------------------------------------------------------
class laplacian(nn.Module):
    def __init__(self, channels):
        super(laplacian, self).__init__()
        laplacian_filter = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        self.conv_x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(torch.from_numpy(laplacian_filter))
        self.conv_y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_y.weight.data.copy_(torch.from_numpy(laplacian_filter.T))
    def forward(self, x):
        laplacianx = self.conv_x(x)
        laplaciany = self.conv_y(x)
        x = torch.abs(laplacianx) + torch.abs(laplaciany)
        return x
#-----------------------------------------------------------------------------------------------------------------------
class Sobelxy(nn.Module):
    def __init__(self, channels):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        # 使用分组卷积对单一特征图全部检测
        self.conv_x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.conv_y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_y.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.conv_x(x)
        sobely = self.conv_y(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x

#-----------------------------------------------------------------------------------------------------------------------
class FMRB(nn.Module):
    def __init__(self, in_feature):
        super(FMRB, self).__init__()
        self.dim_conv3 = in_feature // 4
        self.dim_untouched = in_feature - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.Conv1 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, groups=self.dim_conv3)
        self.Conv2 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=self.dim_conv3)
        self.Conv3 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=5, stride=1, padding=2, dilation=1, bias=True, groups=self.dim_conv3)
        self.Conv4 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=7, stride=1, padding=3, dilation=1, bias=True, groups=self.dim_conv3)
        self.Conv5 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=(1, 3), stride=1, padding=(0,1), dilation=1, bias=True, groups=self.dim_conv3)
        self.Conv6 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=(3, 1), stride=1, padding=(1,0), dilation=1, bias=True, groups=self.dim_conv3)
        self.Sobel = Sobelxy(self.dim_conv3)
        self.Lapla = laplacian(self.dim_conv3)
        self.LRelu = nn.LeakyReLU()
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        out1 = torch.cat((self.Conv1(x1), x2), 1)
        out2 = torch.cat((self.Conv2(x1), x2), 1)
        out3 = torch.cat((self.Conv3(x1), x2), 1)
        out4 = torch.cat((self.Conv4(x1), x2), 1)
        out5 = torch.cat((self.Conv5(x1), x2), 1)
        out6 = torch.cat((self.Conv6(x1), x2), 1)
        out7 = torch.cat((self.Sobel(x1), x2), 1)
        out8 = torch.cat((self.Lapla(x1), x2), 1)
        end = self.LRelu(out1+out2+out3+out4+out5+out6+out7+out8)
        end = torch.add(end, x)
        return end
#-----------------------------------------------------------------------------------------------------------------------
class ECA(nn.Module):
    def __init__(self):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        out = self.avg_pool(y)
        out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return x * out.expand_as(x)
#-----------------------------------------------------------------------------------------------------------------------
class Fast_Conv_Attention(nn.Module):
    def __init__(self, in_feature):
        super(Fast_Conv_Attention, self).__init__()
        self.ERB = FMRB(in_feature)
        self.ECA = ECA()
    def forward(self, x):
        out = self.ERB(x)
        out = self.ECA(x, out)
        return out
#-----------------------------------------------------------------------------------------------------------------------
class Fast_low_light_enhancement(nn.Module):
    def __init__(self, in_feature = 30):
        super(Fast_low_light_enhancement, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels=3, out_channels=in_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.Conv_2_1 = nn.Conv2d(in_channels=in_feature, out_channels=in_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.Conv_2_2 = nn.Conv2d(in_channels=in_feature, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.FCA = Fast_Conv_Attention(in_feature)
        self.transformer = Transformer(in_feature, 1, in_feature, in_feature, dropout=0.)
        self.LRelu = nn.LeakyReLU()
        self.Conv_3 = nn.Conv2d(in_channels=3, out_channels=in_feature, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.Conv_end = nn.Conv2d(in_channels=in_feature, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
    def forward(self, x):
        n, c, h, w = x.shape
        out = self.LRelu(self.Conv_1(x))
        out0 = self.FCA(out)
        out1_f = self.FCA(out0)
        #---------------------------------------------------------------------------------------------------------------
        out1_1 = nn.AdaptiveAvgPool2d((h // 8, w // 8))(out0)
        out1_2 = self.Conv_2_1(out1_1)
        out1_3 = rearrange(out1_2, 'b  c  h  w ->  b (h w)  c ')
        out1_4 = self.transformer(out1_3)
        out1_5 = torch.add(rearrange(out1_4, 'b (h w) c -> b c h w', h=np.sqrt(out1_4.shape[1]).astype(int)), out1_2)
        out1_6 = self.Conv_2_2(out1_5)
        out1_7 = F.upsample_bilinear(out1_6, (h//2, w//2)).tanh()
        out1_8 = F.interpolate(out1_7, out.shape[2:], mode='bilinear')
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(out1_8, 1, dim=1)
        x = x.to(out1_8.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)
        out1_t = torch.clamp(enhanced_image, 0, 1)
        out1_t = self.Conv_3(out1_t)
        #Fast_Conv branch----------------------------------------------------------------------------------------------------
        end = torch.add(out1_f, out1_t)
        end = torch.add(self.FCA(end), out)
        end = self.LRelu(self.Conv_end(end))

        return end



    # def _fuse_bn_tensor(self, branch):
    #     if branch is None:
    #         return 0, 0
    #     if isinstance(branch, nn.Sequential):
    #         kernel = branch.conv.weight
    #         running_mean = branch.bn.running_mean
    #         running_var = branch.bn.running_var
    #         gamma = branch.bn.weight
    #         beta = branch.bn.bias
    #         eps = branch.bn.eps
    #     else:
    #         assert isinstance(branch, nn.BatchNorm2d)
    #         if not hasattr(self, 'id_tensor'):
    #             input_dim = self.in_channels // self.groups
    #             kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
    #             for i in range(self.in_channels):
    #                 kernel_value[i, i % input_dim, 1, 1] = 1
    #             self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
    #         kernel = self.id_tensor
    #         running_mean = branch.running_mean
    #         running_var = branch.running_var
    #         gamma = branch.weight
    #         beta = branch.bias
    #         eps = branch.eps
    #     std = (running_var + eps).sqrt()
    #     t = (gamma / std).reshape(-1, 1, 1, 1)
    #     return kernel * t, beta - running_mean * gamma / std

    # def get_equivalent_kernel_bias(self):
    #     kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
    #     kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
    #     kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
    #     ………………………………
    #     return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid + ………………




