import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from sklearn.metrics import f1_score
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MomentAggregation(nn.Module):
    def __init__(self, order=3):
        super(MomentAggregation, self).__init__()
        self.order = order

    def forward(self, x):
        moments = [x.mean(dim=(2, 3), keepdim=True)]
        if self.order >= 2:
            for i in range(2, self.order + 1):
                moment = torch.mean((x - moments[0]) ** i, dim=(2, 3), keepdim=True)
                moments.append(moment)
        return torch.cat(moments, dim=1)


class CrossMomentConvolution(nn.Module):
    def __init__(self, in_channels, order=3):
        super(CrossMomentConvolution, self).__init__()
        self.order = order
        self.conv = nn.Conv2d(in_channels * order, in_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MCA(nn.Module):  #  SE
    def __init__(self, in_channels, order=3):
        super(MCA, self).__init__()
        self.moment_aggregation = MomentAggregation(order)
        self.cross_moment_conv = CrossMomentConvolution(in_channels, order)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        aggregated_moments = self.moment_aggregation(x)
        cross_moment_features = self.cross_moment_conv(aggregated_moments)
        attention_weights = self.sigmoid(cross_moment_features)
        return x * attention_weights


# 跨层空间注意力
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 跨层通道注意力
# class ChannelAttention(nn.Module): # 12通道的正确代码
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(in_channels // reduction, 1)  # 确保通道数不会为零

        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# class CapsuleNetwork(nn.Module):
#     def __init__(self, num_classes=15):
#         super(CapsuleNetwork, self).__init__()
#         self.conv1 = ConvBlock(12, 32)  # 输入是单通道数据,改成12通道
#         self.mca = MCA(in_channels=32, order=3)
#         self.spatial_attention = SpatialAttention()
#         self.channel_attention = ChannelAttention(in_channels=32)
#
#         # 增加一个池化层来减少数据维度
#         self.pool = nn.AdaptiveAvgPool2d((12, 100))
#
#         # 动态计算全连接层的输入大小
#         self.fc_input_size = 32 * 12 * 100
#
#         self.capsule1 = nn.Linear(self.fc_input_size, 128)
#         self.capsule2 = nn.Linear(128, 256)
#         self.fc = nn.Linear(256, num_classes)
#         # self.sigmoid = nn.Sigmoid()  # 添加sigmoid以确保正输出
#
#     def forward(self, x):
#         # 将数据的形状从 [batch_size, 1, 12, 5000] 调整为 [batch_size, 12,1, 5000]
#         x = x.permute(0, 2, 1, 3)
#         # print('x', x.shape)  # x torch.Size([128, 12, 1, 5000])
#         x = self.conv1(x)
#         x = self.mca(x)
#
#         # 应用空间注意力机制
#         spatial_weights = self.spatial_attention(x)
#         x = x * spatial_weights
#
#         # 应用通道注意力机制
#         channel_weights = self.channel_attention(x)
#         x = x * channel_weights
#
#         # 使用池化层
#         x = self.pool(x)
#
#         x = x.view(x.size(0), -1)  # 展平
#         x = F.relu(self.capsule1(x))
#         x = F.relu(self.capsule2(x))
#         x = self.fc(x)
#         # x = self.sigmoid(x)  # sigmoid activation
#         return x

class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes=15):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = ConvBlock(1, 15)  # 输入是单通道数据,改成12通道
        self.mca = MCA(in_channels=15, order=3)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels=15)

        # 增加一个池化层来减少数据维度
        self.pool = nn.AdaptiveAvgPool2d((12, 100))

        # 动态计算全连接层的输入大小
        self.fc_input_size = 15 * 12 * 100

        self.capsule1 = nn.Linear(self.fc_input_size, 128)
        self.capsule2 = nn.Linear(128, 256)
        self.fc = nn.Linear(256, num_classes)
        # self.sigmoid = nn.Sigmoid()  # 添加sigmoid以确保正输出

    def forward(self, x):
        # 将数据的形状从 [batch_size, 1, 12, 5000] 调整为 [batch_size, 12,1, 5000]
        # x = x.permute(0, 2, 1, 3)
        # print('x', x.shape)  # x torch.Size([128, 12, 1, 5000])
        x = self.conv1(x)
        x = self.mca(x)

        # 应用空间注意力机制
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        # 应用通道注意力机制
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # 使用池化层
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.capsule1(x))
        x = F.relu(self.capsule2(x))
        x = self.fc(x)

        return x


