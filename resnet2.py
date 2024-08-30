import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

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

class MCA(nn.Module):
    def __init__(self, in_channels, order=3):
        super(MCA, self).__init__()
        self.moment_aggregation = MomentAggregation(order)
        self.cross_moment_conv = CrossMomentConvolution(in_channels, order)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        aggregated_moments = self.moment_aggregation(x)
        cross_moment_features = self.cross_moment_conv(aggregated_moments)
        attention_weights = self.sigmoid(cross_moment_features)
        return x * attention_weights

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # print('SpatialAttention input shape:', x.shape)
        x = self.conv1(x)
        # print('SpatialAttention conv output shape:', x.shape)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 检查并打印 reduction 计算后的通道数
        reduced_channels = max(in_channels // reduction, 1)
        # print(f'Original in_channels: {in_channels}, Reduced channels: {reduced_channels}')

        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('ChannelAttention input shape:', x.shape)

        avg_pooled = self.avg_pool(x)
        # print('Average pooled shape:', avg_pooled.shape)

        max_pooled = self.max_pool(x)
        # print('Max pooled shape:', max_pooled.shape)

        fc1_avg = self.fc1(avg_pooled)
        # print('After fc1 (avg pooled) shape:', fc1_avg.shape)

        fc1_max = self.fc1(max_pooled)
        # print('After fc1 (max pooled) shape:', fc1_max.shape)

        relu_avg = self.relu1(fc1_avg)
        # print('After ReLU (avg pooled) shape:', relu_avg.shape)

        relu_max = self.relu1(fc1_max)
        # print('After ReLU (max pooled) shape:', relu_max.shape)

        fc2_avg = self.fc2(relu_avg)
        # print('After fc2 (avg pooled) shape:', fc2_avg.shape)

        fc2_max = self.fc2(relu_max)
        # print('After fc2 (max pooled) shape:', fc2_max.shape)

        out = fc2_avg + fc2_max
        return self.sigmoid(out)




class ResNetAttention(nn.Module):
    def __init__(self, num_classes=15):
        super(ResNetAttention, self).__init__()
        self.mca = MCA(in_channels=12, order=3)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels=12)

        self.resnet = resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # print('Original input shape:', x.shape)
        x = x.permute(0, 2, 1, 3)
        # print('Permuted input shape:', x.shape)

        x = self.mca(x)
        # print('MCA output shape:', x.shape)

        spatial_weights = self.spatial_attention(x)
        x_spatial = x * spatial_weights
        # print('Spatial attention applied shape:', x_spatial.shape)

        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        # print('Channel attention applied shape:', x_channel.shape)

        x = x_spatial + x_channel
        # print('Combined attention shape:', x.shape)

        x = self.resnet(x)  # 这个改成最后一层用于分类
        # print('ResNet output shape:', x.shape)
        x = self.fc(x)
        # print('Final output shape:', x.shape)
        return x


# Original input shape: torch.Size([128, 1, 12, 5000])
# MCA output shape: torch.Size([128, 12, 1, 5000])
# Spatial attention applied shape: torch.Size([128, 12, 1, 5000])
# Channel attention applied shape: torch.Size([128, 12, 1, 5000])
# Combined attention shape: torch.Size([128, 12, 1, 5000])
# ResNet output shape: torch.Size([128, 256])
# Final output shape: torch.Size([128, 15])
from torchinfo import summary

# 假设你的 ResNetAttention 类在 resnet_attention.py 文件中
# from resnet_attention import ResNetAttention

model = ResNetAttention(num_classes=15)
summary(model, input_size=(128, 1, 12, 5000))  # 根据实际的输入形状调整
