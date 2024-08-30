# 定义训练和测试函数
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from torch import optim, nn, device
from ptbxl.zqy.metrics import *
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_curve
import torch.nn.functional as F
from ptbxl.zqy.network import CapsuleNetwork
# from ptbxl.zqy.nopre import train_loader, test_loader

#######################数据增强后的train_loader, test_loader
from ptbxl.resnet_se.multidata import MultiChannelDataset  # 确保你已经定义了这个类
from torch.utils.data import DataLoader
# 加载数据
train_data = np.load('E:/search/daima/ptbxl/resnet_se/train_data.npz')
X_train_augmented = [torch.tensor(train_data[f'X_train_augmented_{i}'], dtype=torch.float32) for i in range(12)]
y_train_augmented = torch.tensor(train_data['y_train_augmented'], dtype=torch.float32)

# test_data = np.load('E:/search/daima/ptbxl/resnet_se/test_data.npz')
# X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
# y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)

######### 使用增强后的测试集，增大数量
test_data = np.load('E:/search/daima/ptbxl/resnet_se/a_test_data.npz')
X_test = [torch.tensor(test_data[f'X_test_augmented_{i}'], dtype=torch.float32) for i in range(12)]
y_test = torch.tensor(test_data['y_test_augmented'], dtype=torch.float32)


train_dataset = MultiChannelDataset(X_train_augmented, y_train_augmented)
test_dataset = MultiChannelDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
###############################################################

# # 从文件加载
# correlation_matrix = torch.load('E:/search/daima/ptbxl/correlation_matrix.pt')
# print(correlation_matrix)

# 计算类别权重
def calculate_class_weights(labels):
    class_counts = labels.sum(axis=0)
    total_samples = labels.shape[0]
    class_weights = total_samples / (class_counts + 1e-5)  # 避免除以0
    return torch.FloatTensor(class_weights).to(device)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签分别加载到设备上
        data = torch.stack(data[:-1], dim=1).unsqueeze(1).to(device)
        target = data[-1].to(device)
        optimizer.zero_grad()
        output = model(data, target)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()


# def train(model, device, train_loader, optimizer, criterion):  #内存不足
#     model.train()
#     scaler = torch.cuda.amp.GradScaler()  # 创建GradScaler以启用混合精度训练
#
#     for batch_samples, batch_labels in train_loader:
#         batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)  # 将样本堆叠到一个张量并增加一个通道维度
#         batch_labels = batch_labels.to(device)
#
#         optimizer.zero_grad()
#
#         # 使用autocast启用混合精度
#         with torch.cuda.amp.autocast():
#             output = model(batch_samples, correlation_matrix)
#             loss = criterion(output, batch_labels)
#
#         # 反向传播和优化步骤
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

def te(model, device, test_loader, criterion, threshold=0.15):
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch_samples, batch_labels in test_loader:
            batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)  # 将样本堆叠到一个张量并增加一个通道维度
            batch_labels = batch_labels.to(device)

            output = model(batch_samples)
            test_loss += criterion(output, batch_labels).item()

            # 将预测结果转换为多标签格式
            preds = (output > threshold).int()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_labels.cpu().numpy())
            all_outputs.append(output.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    test_loss /= len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='samples')  # 使用 samples 进行评价

    # 计算分类报告
    report = classification_report(all_targets, all_preds,
                                   target_names=['AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
                                                 'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI','NORM' ], zero_division=0)
    print(report)

    # 计算准确率
    accuracy = accuracy_score(all_targets, all_preds)

    # 计算AUROC
    try:
        auroc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
    except ValueError:
        auroc = 'AUROC could not be computed.'

    # 计算灵敏度、特异性
    cm = confusion_matrix(all_targets.argmax(axis=1), all_preds.argmax(axis=1))
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # 计算Hamming Loss
    ham_loss = hamming_loss(all_targets, all_preds)

    return test_loss, f1, accuracy, auroc, sensitivity, specificity, ham_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsuleNetwork().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 6.27修改，进行 L2 正则化，防止模型过拟合

# 计算训练数据的类别权重
all_train_labels = []
for _, batch_labels in train_loader:
    all_train_labels.append(batch_labels.cpu().numpy())  # 移动到CPU然后转换为NumPy数组
all_train_labels = np.vstack(all_train_labels)

class_weights = calculate_class_weights(all_train_labels)
# 调整权重策略
adjusted_weights = class_weights ** 0.5 # 使用平方根缩放权重
criterion = nn.BCEWithLogitsLoss(pos_weight=adjusted_weights)  # 使用加权的二元交叉熵损失

# 改进：使用焦点损失！！！
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, pos_weight=None):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.pos_weight = pos_weight
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()
#
# # 计算类别权重
# def calculate_class_weights(labels):
#     class_counts = labels.sum(axis=0)
#     total_samples = labels.shape[0]
#     class_weights = total_samples / (len(class_counts) * class_counts)
#     return torch.tensor(class_weights, dtype=torch.float32).to(device)
#
# all_train_labels = []
# for _, batch_labels in train_loader:
#     all_train_labels.append(batch_labels.cpu().numpy())  # 移动到CPU然后转换为NumPy数组
# all_train_labels = np.vstack(all_train_labels)
#
# class_weights = calculate_class_weights(all_train_labels)
# # # 使用Focal Loss
# # criterion = FocalLoss(alpha=1, gamma=2, pos_weight=class_weights)

num_epochs = 15  # 100
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, criterion)

    train_loss, train_f1, train_accuracy, train_auroc, train_sensitivity, train_specificity, train_ham_loss = te(model, device, train_loader, criterion)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, Train AUROC: {train_auroc}, '
          f'Train Sensitivity: {train_sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, '
          f'Train Hamming Loss: {train_ham_loss:.4f}')
    print('----------------------------------------------------------------------------------------------------------------------------------')
    test_loss, test_f1, test_accuracy, test_auroc, test_sensitivity, test_specificity, test_ham_loss = te(model, device, test_loader, criterion)

    print(f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auroc}, '
          f'Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_specificity:.4f}, '
          f'Test Hamming Loss: {test_ham_loss:.4f}')