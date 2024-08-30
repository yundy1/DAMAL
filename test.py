# 定义训练和测试函数
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from torch import optim, nn, device

# from ptbxl.resnet_se.no_ca import ResNetAttention
# from ptbxl.resnet_se.no_sa import ResNetAttention  # 无空间注意力机制
from ptbxl.zqy.metrics import *
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_curve

# from ptbxl.zqy.model_shunxu import ResNetAttention
from ptbxl.zqy.network import CapsuleNetwork
from ptbxl.zqy.pre import labels_train
import torch

# from ptbxl.zqy.resnet2 import ResNetAttention

# from ptbxl.resnet_se.no_mca import ResNetAttentionWithoutMCA  # 无mca

torch.cuda.empty_cache()


# 计算类别权重
def calculate_class_weights(labels):
    class_counts = labels.sum(axis=0)
    total_samples = labels.shape[0]
    class_weights = total_samples / (class_counts + 1e-5)  # 避免除以0
    return torch.FloatTensor(class_weights).to(device)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_samples, batch_labels in train_loader:
        batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)  # 将样本堆叠到一个张量并增加一个通道维度
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_samples)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()


# def te(model, device, test_loader, criterion, threshold=0.3):
#     model.eval()
#     test_loss = 0
#     all_preds = []
#     all_targets = []
#     all_outputs = []
#
#     with torch.no_grad():
#         for batch_samples, batch_labels in test_loader:
#             batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)  # 将样本堆叠到一个张量并增加一个通道维度
#             batch_labels = batch_labels.to(device)
#
#             output = model(batch_samples)
#             test_loss += criterion(output, batch_labels).item()
#
#             # 将预测结果转换为多标签格式
#             preds = (output > threshold).int()
#
#             all_preds.append(preds.cpu().numpy())
#             all_targets.append(batch_labels.cpu().numpy())
#             all_outputs.append(output.cpu().numpy())
#
#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)
#     all_outputs = np.vstack(all_outputs)
#
#     test_loss /= len(test_loader.dataset)
#     f1 = f1_score(all_targets, all_preds, average='samples')  # 使用 samples 进行评价
#
#     # 计算分类报告
#     report = classification_report(all_targets, all_preds,
#                                    target_names=['AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
#                                                  'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI','NORM' ], zero_division=0)
#     # report = classification_report(all_targets, all_preds,
#     #                                target_names=['IMI', 'AMI', 'LMI', 'other','NORM' ], zero_division=0)
#     print(report)
#
#     # 计算准确率
#     accuracy = accuracy_score(all_targets, all_preds)
#
#     # 计算AUROC
#     try:
#         auroc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
#     except ValueError:
#         auroc = 'AUROC could not be computed.'
#
#     # 计算灵敏度、特异性
#     cm = confusion_matrix(all_targets.argmax(axis=1), all_preds.argmax(axis=1))
#     tn = cm[0, 0]
#     tp = cm[1, 1]
#     fn = cm[1, 0]
#     fp = cm[0, 1]
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#
#     return test_loss, f1, accuracy, auroc, sensitivity, specificity


# 自定义的  CPU版本  #########################################################################
# def find_best_thresholds(model, device, val_loader):
#     model.eval()
#     all_outputs = []
#     all_targets = []
#
#     with torch.no_grad():
#         for batch_samples, batch_labels in val_loader:
#             batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)
#             batch_labels = batch_labels.to(device)
#
#             output = model(batch_samples)
#
#             all_outputs.append(output.cpu().numpy())
#             all_targets.append(batch_labels.cpu().numpy())
#
#     all_outputs = np.vstack(all_outputs)
#     all_targets = np.vstack(all_targets)
#
#     best_thresholds = []
#     for i in range(all_targets.shape[1]):
#         fpr, tpr, thresholds = roc_curve(all_targets[:, i], all_outputs[:, i])
#         youdens_j = tpr - fpr
#         best_threshold = thresholds[np.argmax(youdens_j)]
#         best_thresholds.append(best_threshold)
#
#     return np.array(best_thresholds)
#
# def evaluate(model, device, loader, criterion, thresholds):
#     model.eval()
#     eval_loss = 0
#     all_preds = []
#     all_targets = []
#     all_outputs = []
#
#     with torch.no_grad():
#         for batch_samples, batch_labels in loader:
#             batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)
#             batch_labels = batch_labels.to(device)
#
#             output = model(batch_samples)
#             eval_loss += criterion(output, batch_labels).item()
#
#             preds = np.zeros_like(output.cpu().numpy())
#             for i in range(output.shape[1]):
#                 preds[:, i] = (output[:, i] > thresholds[i]).int()
#
#             all_preds.append(preds)
#             all_targets.append(batch_labels.cpu().numpy())
#             all_outputs.append(output.cpu().numpy())
#
#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)
#     all_outputs = np.vstack(all_outputs)
#
#     eval_loss /= len(loader.dataset)
#
#     accuracy = accuracy_score(all_targets, all_preds)
#
#     # 计算分类报告
#     target_names = ['AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
#                     'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI', 'NORM']
#     report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
#
#     print("Precision, Recall, F1-score per class:")
#     for class_name, metrics in report.items():
#         if class_name in target_names:
#             print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']}")
#
#     print(f"Macro avg: Precision: {report['macro avg']['precision']:.4f}, Recall: {report['macro avg']['recall']:.4f}, F1-score: {report['macro avg']['f1-score']:.4f}")
#     print(f"Micro avg: Precision: {report['micro avg']['precision']:.4f}, Recall: {report['micro avg']['recall']:.4f}, F1-score: {report['micro avg']['f1-score']:.4f}")
#     print(f"Weighted avg: Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1-score: {report['weighted avg']['f1-score']:.4f}")
#     print(f"Samples avg F1-score: {f1_score(all_targets, all_preds, average='samples'):.4f}")
#
#     # 计算AUROC
#     try:
#         auroc = roc_auc_score(all_targets, all_outputs, average='macro')
#     except ValueError:
#         auroc = 'AUROC could not be computed.'
#
#     return eval_loss, report, auroc, accuracy
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = CapsuleNetwork().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# class_weights = calculate_class_weights(labels_train)
# adjusted_weights = class_weights ** 0.5  # 使用平方根缩放权重
# criterion = nn.BCEWithLogitsLoss(pos_weight=adjusted_weights)
#
# num_epochs = 100
# for epoch in range(num_epochs):
#     train(model, device, train_loader, optimizer, criterion)
#
#     # 在验证集上评估
#     thresholds = find_best_thresholds(model, device, val_loader)
#     print(f'Epoch {epoch + 1}, Best thresholds: {thresholds}')
#
#     val_loss, val_report, val_auc, val_accuracy = evaluate(model, device, val_loader, criterion, thresholds)
#     print(
#         f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUROC: {val_auc}')
#
# # 在测试集上评估模型性能
# test_loss, test_report, test_auc, test_accuracy = evaluate(model, device, test_loader, criterion, thresholds)
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auc}')
# print("Test Classification Report:")
# print(test_report)


def find_best_thresholds(model, device, val_loader, criterion):
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_samples)
            all_outputs.append(output.cpu().numpy())
            all_targets.append(batch_labels.cpu().numpy())

    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    best_thresholds = []
    for i in range(all_targets.shape[1]):
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.linspace(0, 1, 101):
            preds = (all_outputs[:, i] > threshold).astype(int)
            f1 = f1_score(all_targets[:, i], preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        best_thresholds.append(best_threshold)

    return best_thresholds


# def evaluate(model, device, loader, criterion, thresholds):
#     model.eval()
#     eval_loss = 0
#     all_preds = []
#     all_targets = []
#     all_outputs = []
#
#     with torch.no_grad():
#         for batch_samples, batch_labels in loader:
#             batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)
#             batch_labels = batch_labels.to(device)
#
#             output = model(batch_samples)
#             eval_loss += criterion(output, batch_labels).item()
#
#             output_cpu = output.cpu().numpy()  # 将输出转换为 CPU 张量
#
#             preds = np.zeros_like(output_cpu)
#             for i in range(output_cpu.shape[1]):
#                 preds[:, i] = (output_cpu[:, i] > thresholds[i]).astype(int)  # 使用 CPU 张量
#
#             all_preds.append(preds)
#             all_targets.append(batch_labels.cpu().numpy())
#             all_outputs.append(output_cpu)
#
#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)
#     all_outputs = np.vstack(all_outputs)
#
#     eval_loss /= len(loader.dataset)
#
#     accuracy = accuracy_score(all_targets, all_preds)
#     f1 = f1_score(all_targets, all_preds, average='samples')  # 使用 samples 进行评价
#     # 计算分类报告
#     report = classification_report(all_targets, all_preds,
#                                    target_names=['AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
#                                                  'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI', 'NORM'],
#                                    zero_division=0)
#
#     print(report)
#
#
#     # 计算AUROC
#     try:
#         auroc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
#     except ValueError:
#         auroc = 'AUROC could not be computed.'
#
#     # 计算灵敏度、特异性
#     cm = confusion_matrix(all_targets.argmax(axis=1), all_preds.argmax(axis=1))
#     tn = cm[0, 0]
#     tp = cm[1, 1]
#     fn = cm[1, 0]
#     fp = cm[0, 1]
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#
#     return eval_loss, f1, accuracy, auroc, sensitivity, specificity

# 修改代码
def te(model, device, test_loader, criterion, thresholds):
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch_samples, batch_labels in test_loader:
            batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_samples)
            test_loss += criterion(output, batch_labels).item()

            # 将预测结果转换为多标签格式
            preds = (torch.sigmoid(output) > torch.tensor(thresholds).to(device)).int()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_labels.cpu().numpy())
            all_outputs.append(output.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    test_loss /= len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='samples')

    # # 计算分类报告
    report = classification_report(all_targets, all_preds,
                                   target_names=['AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
                                                 'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI','NORM' ], zero_division=0)
    #
    print(report)

    reportdict = classification_report(
        all_targets, all_preds, target_names=[
            'AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
            'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI', 'NORM'
        ], zero_division=0, output_dict=True)

    # 按类别逐行打印分类报告
    for class_name, metrics in reportdict.items():
        if class_name in ['macro avg', 'weighted avg', 'samples avg', 'micro avg']:
            print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                  f"F1-score: {metrics['f1-score']:.4f}")
        else:
            print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                  f"F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']}")

    # 计算准确率
    accuracy = accuracy_score(all_targets, all_preds)

    # 计算AUROC
    # try:
    #     auroc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
    # except ValueError:
    #     auroc = 'AUROC could not be computed.'
    # 计算AUROC
    auroc_per_class = []
    for i in range(all_targets.shape[1]):
        try:
            if len(np.unique(all_targets[:, i])) > 1:  # 确保该类不是全0或全1
                auroc_per_class.append(roc_auc_score(all_targets[:, i], all_outputs[:, i]))
            else:
                auroc_per_class.append(np.nan)  # 对于无法计算的类，设置为nan
        except ValueError:
            auroc_per_class.append(np.nan)  # 捕获可能的异常，并设置为nan

    mean_auroc = np.nanmean(auroc_per_class)  # 计算平均AUROC，忽略nan值

    # 计算灵敏度、特异性

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

    return test_loss, f1, accuracy, mean_auroc, sensitivity, specificity, ham_loss


    # print("Precision, Recall, F1-score per class:")
    # for class_name, metrics in report.items():
    #     if class_name in target_names:
    #         print(
    #             f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']}")
    #
    # print(
    #     f"Macro avg: Precision: {report['macro avg']['precision']:.4f}, Recall: {report['macro avg']['recall']:.4f}, F1-score: {report['macro avg']['f1-score']:.4f}")
    # print(
    #     f"Micro avg: Precision: {report['micro avg']['precision']:.4f}, Recall: {report['micro avg']['recall']:.4f}, F1-score: {report['micro avg']['f1-score']:.4f}")
    # print(
    #     f"Weighted avg: Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1-score: {report['weighted avg']['f1-score']:.4f}")
    # print(f"Samples avg F1-score: {f1_score(all_targets, all_preds, average='samples'):.4f}")
    #
    # # 计算AUROC
    # try:
    #     auroc = roc_auc_score(all_targets, all_outputs, average='macro')
    # except ValueError:
    #     auroc = 'AUROC could not be computed.'
    #
    # return eval_loss, report, auroc, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CapsuleNetwork().to(device)


# model = ResNetAttention().to(device)
# 假设模型已经训练好
# 保存模型
torch.save(model.state_dict(), 'capsule_network_629.pth')
# torch.save(model.state_dict(), 'ResNetAttention_threshold2.pth')

optimizer = optim.Adam(model.parameters(), lr=0.001)
class_weights = calculate_class_weights(labels_train)
adjusted_weights = class_weights ** 0.5  # 使用平方根缩放权重
criterion = nn.BCEWithLogitsLoss(pos_weight=adjusted_weights)

num_epochs = 30
# 加载数据
train_data = np.load('E:/search/daima/ptbxl/zqy/train_data1.npz')
X_train_augmented = [torch.tensor(train_data[f'X_train_augmented_{i}'], dtype=torch.float32) for i in range(12)]
y_train_augmented = torch.tensor(train_data['y_train_augmented'], dtype=torch.float32)

val_data = np.load('E:/search/daima/ptbxl/zqy/val_data1.npz')
X_val = [torch.tensor(val_data[f'X_val_{i}'], dtype=torch.float32) for i in range(12)]
y_val = torch.tensor(val_data['y_val'], dtype=torch.float32)

######### 使用原始测试集
# 不区分验证集和测试集，加大数据
test_data = np.load('E:/search/daima/ptbxl/zqy/test_data1.npz')
X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)

# 合并特征
X_combined = [torch.cat((X_val[i], X_test[i]), dim=0) for i in range(12)]

# 合并标签
y_combined = torch.cat((y_val, y_test), dim=0)

# 检查合并后的形状
for i, x in enumerate(X_combined):
    print(f'X_combined[{i}].shape: {x.shape}')
print(f'y_combined.shape: {y_combined.shape}')

# # 保存合并后的数据 (如果需要保存)
# np.savez('E:/search/daima/ptbxl/zqy/combined_data2.npz',
#          **{f'X_combined_{i}': X_combined[i].numpy() for i in range(12)},
#          y_combined=y_combined.numpy())

from ptbxl.resnet_se.multidata import MultiChannelDataset
from torch.utils.data import DataLoader

train_dataset = MultiChannelDataset(X_train_augmented, y_train_augmented)
# test_dataset = MultiChannelDataset(X_test, y_test)
# val_dataset = MultiChannelDataset(X_val, y_val)
test_dataset = MultiChannelDataset(X_combined, y_combined)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# for epoch in range(num_epochs):
#     train(model, device, train_loader, optimizer, criterion)
#
#     # 在验证集上评估
#     thresholds = find_best_thresholds(model, device, val_loader)
#     print(f'Epoch {epoch + 1}, Best thresholds: {thresholds}')
#
#     val_loss, val_report, val_auc, val_accuracy = evaluate(model, device, val_loader, criterion, thresholds)
#     print(
#         f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUROC: {val_auc}')
#     # print(
#     #     f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auroc}, Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_specificity:.4f}')


#
# for epoch in range(num_epochs):
#     train(model, device, train_loader, optimizer, criterion)
#     thresholds = find_best_thresholds(model, device, test_loader)     # 找阈值val_loader
#     print(f'Epoch {epoch + 1}, Best thresholds: {thresholds}')
#     test_loss, test_f1, test_accuracy, test_auroc, test_sensitivity, test_specificity = evaluate(model, device, test_loader,
#     criterion, thresholds)
#     print(f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, '
#           f'Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auroc}, '
#           f'Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_specificity:.4f}')

best_thresholds = [0.44, 0.78, 0.9, 0.63, 0.79, 0.72, 0.13, 0.5, 0.5, 0.6, 0.76, 0.5, 0.53, 0.5, 0.49]

for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, criterion)

    # 在验证集上找到最佳阈值
    # best_thresholds = find_best_thresholds(model, device, val_loader, criterion)
    # print(best_thresholds)
    train_loss, train_f1, train_accuracy, train_auroc, train_sensitivity, train_specificity, train_ham_loss = te(model, device, train_loader, criterion, best_thresholds)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, Train AUROC: {train_auroc}, '
          f'Train Sensitivity: {train_sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, '
          f'Train Hamming Loss: {train_ham_loss:.4f}')
    print('----------------------------------------------------------------------------------------------------------------------------------')

    test_loss, test_f1, test_accuracy, test_auroc, test_sensitivity, test_specificity, test_ham_loss = te(model, device, test_loader, criterion, best_thresholds)

    print(f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auroc}, '
          f'Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_specificity:.4f}, '
          f'Test Hamming Loss: {test_ham_loss:.4f}')

# # 在测试集上评估模型性能
# test_loss, test_report, test_auc, test_accuracy = evaluate(model, device, test_loader, criterion, thresholds)
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUROC: {test_auc}')
# print("Test Classification Report:")
# print(test_report)