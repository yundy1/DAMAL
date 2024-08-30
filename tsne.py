import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

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
target_layer = 'capsule1'  # capsule1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你已经有模型和数据
# 加载模型
model = CapsuleNetwork().to(device)
model.load_state_dict(torch.load('capsule_network.pth'))
model.eval()

# 创建FeatureExtractor
feature_extractor = FeatureExtractor(model, target_layer).to(device)

# 加载数据
# test_data = np.load('E:/search/daima/ptbxl/resnet_se/a_test_data.npz')
# X_test = [torch.tensor(test_data[f'X_test_augmented_{i}'], dtype=torch.float32) for i in range(12)]
# y_test = torch.tensor(test_data['y_test_augmented'], dtype=torch.float32)

# fold_num = 10  # 改变折数,不影响随便
# X_train, y_train, X_test, y_test = read_ptbxl(15, fold_num)
# 加载原始测试集
test_data = np.load('E:/search/daima/ptbxl/zqy/origin_test_data.npz')
X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)
# np.savez('origin_test_data.npz',
#          X_test_0=X_test[0],
#          X_test_1=X_test[1],
#          X_test_2=X_test[2],
#          X_test_3=X_test[3],
#          X_test_4=X_test[4],
#          X_test_5=X_test[5],
#          X_test_6=X_test[6],
#          X_test_7=X_test[7],
#          X_test_8=X_test[8],
#          X_test_9=X_test[9],
#          X_test_10=X_test[10],
#          X_test_11=X_test[11],
#          y_test=y_test.numpy())

# train_data = np.load('E:/search/daima/ptbxl/resnet_se/train_data.npz')
# X_train_augmented = [torch.tensor(train_data[f'X_train_augmented_{i}'], dtype=torch.float32) for i in range(12)]
# y_train_augmented = torch.tensor(train_data['y_train_augmented'], dtype=torch.float32)


# train_data = np.load('E:/search/daima/ptbxl/resnet_se/train_data.npz')
# X_train_augmented = [torch.tensor(train_data[f'X_train_augmented_{i}'], dtype=torch.float32) for i in range(12)]
# y_train_augmented = torch.tensor(train_data['y_train_augmented'], dtype=torch.float32)

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

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42, perplexity=100, learning_rate=200)  # 调整t-SNE参数
tsne_features = tsne.fit_transform(features)


# 可视化
plt.figure(figsize=(10, 8))
for i in range(labels.shape[1]):
    idx = labels[:, i] == 1
    plt.scatter(tsne_features[idx, 0], tsne_features[idx, 1], label=f'Label {i}')
plt.legend()
plt.title('t-SNE of CapsuleNetwork Features')
plt.savefig('tsne_visualization.png')  # 保存图像
plt.show()
