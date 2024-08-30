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
        batch_size, channels = x.size()
        x = x.view(batch_size, channels, 1, 1)
        aggregated_moments = self.moment_aggregation(x)
        cross_moment_features = self.cross_moment_conv(aggregated_moments)
        attention_weights = self.sigmoid(cross_moment_features)
        return x * attention_weights.view(batch_size, channels)

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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ResNetAttention(nn.Module):
    def __init__(self, num_classes=15):
        super(ResNetAttention, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

        self.mca = MCA(in_channels=256, order=3)
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels=256)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.resnet(x)
        x = self.mca(x)

        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.fc(x)
        return x

class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_extractor, num_classes=15):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes

    def forward(self, support, query, support_labels):
        support_features = self.feature_extractor(support)
        query_features = self.feature_extractor(query)

        prototypes = []
        for i in range(self.num_classes):
            class_features = support_features[support_labels[:, i] == 1]
            if len(class_features) == 0:
                prototypes.append(torch.zeros(support_features.size(1), device=support_features.device))
            else:
                class_prototype = class_features.mean(dim=0)
                prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes)

        distances = torch.cdist(query_features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        return distances
