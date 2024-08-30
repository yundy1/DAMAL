# from torch import nn
# import umap
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from ptbxl.zqy.network import CapsuleNetwork
#
# class FeatureExtractor(nn.Module):
#     def __init__(self, model, target_layer):
#         super(FeatureExtractor, self).__init__()
#         self.model = model
#         self.target_layer = target_layer
#         self.feature = None
#
#         self._register_hook()
#
#     def _register_hook(self):
#         def hook(module, input, output):
#             self.feature = output
#
#         layer = dict(self.model.named_modules())[self.target_layer]
#         layer.register_forward_hook(hook)
#
#     def forward(self, x):
#         _ = self.model(x)
#         return self.feature
#
#
# # 选择模型中的目标层
# target_layer = 'capsule2'  # capsule1
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 假设你已经有模型和数据
# # 加载模型
# model = CapsuleNetwork().to(device)
# model.load_state_dict(torch.load('capsule_network.pth'))
# model.eval()
#
# # 创建FeatureExtractor
# feature_extractor = FeatureExtractor(model, target_layer).to(device)
#
# # 加载数据
# test_data = np.load('E:/search/daima/ptbxl/zqy/origin_test_data.npz')
# X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
# y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)
#
# # 创建DataLoader
# test_dataset = TensorDataset(*X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
#
# # 提取特征
# features = []
# labels = []
#
# with torch.no_grad():
#     for data in test_loader:
#         inputs = torch.stack(data[:-1], dim=1).unsqueeze(1).to(device)
#         targets = data[-1].to(device)
#
#         outputs = feature_extractor(inputs)
#         features.append(outputs.cpu().numpy())
#         labels.append(targets.cpu().numpy())
#
# features = np.concatenate(features, axis=0)
# labels = np.concatenate(labels, axis=0)
#
# # 使用UMAP进行降维
# umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
# umap_features = umap_reducer.fit_transform(features)
#
# # 可视化
# plt.figure(figsize=(10, 8))
# for i in range(labels.shape[1]):
#     idx = labels[:, i] == 1
#     plt.scatter(umap_features[idx, 0], umap_features[idx, 1], label=f'Label {i}')
# plt.legend()
# plt.title('UMAP of CapsuleNetwork Features')
# plt.savefig('umap_visualization.png')  # 保存图像
# plt.show()


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import umap.umap_ as umap

from ptbxl.zqy.network import CapsuleNetwork
from ptbxl.zqy.readdata import read_ptbxl


class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layer):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.feature = None

        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.feature = output

        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(hook)

    def forward(self, x):
        _ = self.model(x)
        return self.feature


# 选择模型中的目标层
target_layer = 'capsule2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = CapsuleNetwork().to(device)
model.load_state_dict(torch.load('capsule_network.pth'))
model.eval()

# 创建FeatureExtractor
feature_extractor = FeatureExtractor(model, target_layer).to(device)

# 加载数据
test_data = np.load('E:/search/daima/ptbxl/zqy/origin_test_data.npz')
X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)

# 创建DataLoader
test_dataset = TensorDataset(*X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 提取特征
features = []
labels = []

with torch.no_grad():
    for data in test_loader:
        inputs = torch.stack(data[:-1], dim=1).unsqueeze(1).to(device)
        targets = data[-1].to(device)

        outputs = feature_extractor(inputs)
        features.append(outputs.cpu().numpy())
        labels.append(targets.cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# 对每个标签分别进行降维和可视化
for i in range(labels.shape[1]):
    binary_labels = labels[:, i]

    # 使用UMAP进行降维
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_features = umap_reducer.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_features[binary_labels == 0, 0], umap_features[binary_labels == 0, 1], label='Negative', alpha=0.5)
    plt.scatter(umap_features[binary_labels == 1, 0], umap_features[binary_labels == 1, 1], label='Positive', alpha=0.5)
    plt.legend()
    plt.title(f'UMAP of CapsuleNetwork Features for Label {i}')
    plt.savefig(f'umap_visualization_label_{i}.png')  # 保存图像
    plt.show()


