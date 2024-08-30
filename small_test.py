import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ptbxl.zqy.add_small import PrototypicalNetwork, ResNetAttention


class MultiChannelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[i][idx] for i in range(len(self.X))], self.y[idx]

# 数据准备
train_data = np.load('E:/search/daima/ptbxl/zqy/train_data1.npz')
X_train_augmented = [torch.tensor(train_data[f'X_train_augmented_{i}'], dtype=torch.float32) for i in range(12)]
y_train_augmented = torch.tensor(train_data['y_train_augmented'], dtype=torch.float32)

val_data = np.load('E:/search/daima/ptbxl/zqy/val_data1.npz')
X_val = [torch.tensor(val_data[f'X_val_{i}'], dtype=torch.float32) for i in range(12)]
y_val = torch.tensor(val_data['y_val'], dtype=torch.float32)

test_data = np.load('E:/search/daima/ptbxl/zqy/test_data1.npz')
X_test = [torch.tensor(test_data[f'X_test_{i}'], dtype=torch.float32) for i in range(12)]
y_test = torch.tensor(test_data['y_test'], dtype=torch.float32)

# 合并验证集和测试集
X_combined = [torch.cat((X_val[i], X_test[i]), dim=0) for i in range(12)]
y_combined = torch.cat((y_val, y_test), dim=0)

train_dataset = MultiChannelDataset(X_train_augmented, y_train_augmented)
test_dataset = MultiChannelDataset(X_combined, y_combined)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, hamming_loss, \
    confusion_matrix


def calculate_class_weights(labels):
    class_counts = labels.sum(axis=0)
    total_samples = labels.shape[0]
    class_weights = total_samples / (class_counts + 1e-5)
    return torch.FloatTensor(class_weights).to(device)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_samples, batch_labels in train_loader:
        batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1).to(device)
        batch_labels = batch_labels.to(device)

        # 将数据拆分为support和query集
        support, query = batch_samples[:batch_samples.size(0) // 2], batch_samples[batch_samples.size(0) // 2:]
        support_labels, query_labels = batch_labels[:batch_labels.size(0) // 2], batch_labels[batch_labels.size(0) // 2:]

        optimizer.zero_grad()
        output = model(support, query, support_labels)
        loss = criterion(output, query_labels)
        loss.backward()
        optimizer.step()

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

            support, query = batch_samples[:batch_samples.size(0) // 2], batch_samples[batch_samples.size(0) // 2:]
            support_labels, query_labels = batch_labels[:batch_labels.size(0) // 2], batch_labels[batch_labels.size(0) // 2:]

            output = model(support, query, support_labels)
            test_loss += criterion(output, query_labels).item()

            preds = (torch.sigmoid(output) > torch.tensor(thresholds).to(device)).int()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(query_labels.cpu().numpy())
            all_outputs.append(output.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    test_loss /= len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='samples')

    report = classification_report(
        all_targets, all_preds, target_names=[
            'AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'IPLMI', 'IPMI', 'INJIN',
            'INJIL', 'INJAS', 'INJAL', 'INJLA', 'LMI', 'PMI', 'NORM'
        ], zero_division=0, output_dict=True)

    # 按类别逐行打印分类报告
    for class_name, metrics in report.items():
        if class_name in ['macro avg', 'weighted avg', 'samples avg', 'micro avg']:
            continue
        print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1-score: {metrics['f1-score']:.4f}, Support: {metrics['support']}")

    accuracy = accuracy_score(all_targets, all_preds)

    auroc_per_class = []
    for i in range(all_targets.shape[1]):
        try:
            if len(np.unique(all_targets[:, i])) > 1:
                auroc_per_class.append(roc_auc_score(all_targets[:, i], all_outputs[:, i]))
            else:
                auroc_per_class.append(np.nan)
        except ValueError:
            auroc_per_class.append(np.nan)

    mean_auroc = np.nanmean(auroc_per_class)

    cm = confusion_matrix(all_targets.argmax(axis=1), all_preds.argmax(axis=1))
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    ham_loss = hamming_loss(all_targets, all_preds)

    return test_loss, f1, accuracy, mean_auroc, sensitivity, specificity, ham_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PrototypicalNetwork(feature_extractor=ResNetAttention(num_classes=15)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
class_weights = calculate_class_weights(y_train_augmented.numpy())
adjusted_weights = class_weights ** 0.5
criterion = nn.BCEWithLogitsLoss(pos_weight=adjusted_weights)

best_thresholds = [0.44, 0.78, 0.9, 0.63, 0.79, 0.72, 0.5, 0.5, 0.5, 0.6, 0.76, 0.5, 0.53, 0.5, 0.49]

for epoch in range(10):
    train(model, device, train_loader, optimizer, criterion)
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
