import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

from mmdet3d.models.builder import BACKBONES
import pdb
from debug.utils import print_detail as pd, mem, save_feature_map_as_image, count_trainable_parameters as param


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(nn.BatchNorm2d(in_ch),
                                          double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation))
            else:
                self.conv = nn.Sequential(nn.BatchNorm2d(in_ch), double_conv(in_ch, out_ch, group_conv=False, dilation=dilation))
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
            else:
                self.conv = double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):

        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)), nn.BatchNorm2d(out_ch),
                                      nn.LeakyReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                                      nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                                       nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch), nn.BatchNorm2d(out_ch),
                                       nn.LeakyReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)), nn.BatchNorm2d(out_ch),
                                       nn.LeakyReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)), nn.BatchNorm2d(out_ch),
                                       nn.LeakyReLU(inplace=True))

    def forward(self, x):
        #add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                        double_conv_circular(in_ch, out_ch, group_conv=group_conv, dilation=dilation))
        else:
            self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock=False, drop_p=0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, groups=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch, group_conv=group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv=group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class myMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(myMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.attention_radar = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(input_dim, output_dim, kernel_size=1),
                                             nn.Sigmoid())

    def forward(self, x):
        # Reshape the input tensor to (B, C*H*W) before passing it through the MLP
        attn_radar = self.attention_radar(x)
        B, C, H, W = x.size()
        # breakpoint()
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        return self.mlp(x).reshape(B, H, W, C).permute(0, 3, 1, 2), attn_radar


@BACKBONES.register_module()
class TPVUNet(nn.Module):

    def __init__(self,
                 dim=128,
                 dilation=1,
                 bilinear=True,
                 group_conv=False,
                 input_batch_norm=True,
                 channels=[128, 256, 512],
                 dropout=0.5,
                 circular_padding=False,
                 dropblock=False,
                 light=False,
                 use_feature_distillation=False,
                 **kwargs):
        super().__init__()

        self.use_feature_distillation = use_feature_distillation
        if self.use_feature_distillation:
            self.distill_mlp = nn.ModuleList([myMLP(channels[i], channels[i], channels[i]) for i in range(3)])

        self.inc = inconv(dim, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024,
                      512,
                      circular_padding,
                      bilinear=bilinear,
                      group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout)
        self.up2 = up(768,
                      256,
                      circular_padding,
                      bilinear=bilinear,
                      group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout)
        self.up3 = up(384,
                      128,
                      circular_padding,
                      bilinear=bilinear,
                      group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout)
        self.up4 = up(192,
                      128,
                      circular_padding,
                      bilinear=bilinear,
                      group_conv=group_conv,
                      use_dropblock=dropblock,
                      drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        self.outc = outconv(128, dim)

    def forward(self, tpv_list):

        out_tpv_list = []
        out_tpv_feature_lists = []
        out_tpv_residue_lists = []

        for i in range(len(tpv_list)):
            x = tpv_list[i]

            x1 = self.inc(x)  # [B, 64, 128, 128]
            x2 = self.down1(x1)  # [B, 128, 64, 64]

            if self.use_feature_distillation:
                x2_residue, x2_atten = self.distill_mlp[0](x2)
                x2 = torch.add(x2, torch.mul(x2_residue, x2_atten))

            x3 = self.down2(x2)  # [B, 256, 32, 32]
            if self.use_feature_distillation:
                x3_residue, x3_atten = self.distill_mlp[1](x3)
                x3 = torch.add(x3, torch.mul(x3_residue, x3_atten))
            x4 = self.down3(x3)  # [B, 512, 16, 16]
            if self.use_feature_distillation:
                x4_residue, x4_atten = self.distill_mlp[2](x4)
                x4 = torch.add(x4, torch.mul(x4_residue, x4_atten))
            x5 = self.down4(x4)  # [B, 512, 16, 16]

            x = self.up1(x5, x4)  # 512, 512
            x = self.up2(x, x3)  # 512, 256
            x = self.up3(x, x2)  # 256, 128
            x = self.up4(x, x1)  # 128, 64
            x = self.outc(self.dropout(x))

            out_tpv_list.append(x)
            if self.use_feature_distillation:
                out_feature_list = [x, x2, x3, x4, x5]
                out_residue_list = [x2_residue, x3_residue, x4_residue]
                out_tpv_feature_lists.append(out_feature_list)
                out_tpv_residue_lists.append(out_residue_list)

        if self.use_feature_distillation:
            return out_tpv_list, out_tpv_feature_lists, out_tpv_residue_lists
        else:
            return out_tpv_list, out_tpv_feature_lists, None
