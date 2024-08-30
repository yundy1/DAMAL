import numpy as np
from torch.utils.data import DataLoader

from readdata import *
import torch
from datasets import matDataset

def prepare_trte_data(data_folder, num_class, num_fold, num_view=12):  #12导联 num_view
    if 'PTBXL' in data_folder:
        data_tr_list, labels_tr, data_te_list, labels_te = read_ptbxl(num_class, num_fold)


    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    data_mat_list = []
    for i in range(num_view):
        # 分别处理12导联数据
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))

    data_tensor_list = [torch.FloatTensor(mat) for mat in data_mat_list]

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, num_tr + num_te))
    data_train_list = [tensor[idx_dict["tr"]] for tensor in data_tensor_list]
    data_test_list = [tensor[idx_dict["te"]] for tensor in data_tensor_list]

    labels = np.concatenate((labels_tr, labels_te))

    # 打印数据类型和形状
    print("Data Train List:")
    for i, tensor in enumerate(data_train_list):
        print(f"View {i + 1}: Type={type(tensor)}, Shape={tensor.shape}") #View 12: Type=<class 'torch.Tensor'>, Shape=torch.Size([1513, 5000])

    print("Data Test List:")
    for i, tensor in enumerate(data_test_list):
        print(f"View {i + 1}: Type={type(tensor)}, Shape={tensor.shape}")

    print(f"Labels: Type={type(labels)}, Shape={labels.shape}") #Labels: Type=<class 'numpy.ndarray'>, Shape=(14982, 15)

    return data_train_list, data_test_list, idx_dict, labels



dataset='PTBXL'
fold_num = 10  #改变折数,不影响随便
data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(dataset, 15, fold_num, num_view=12)
#data_tr_list 训练集 X;data_test_list 测试集 X;trte_idx 索引；labels_trte  所有 y

labels_tr_tensor = torch.FloatTensor(labels_trte[trte_idx["tr"]]) #训练集 y
labels_te_tensor = torch.FloatTensor(labels_trte[trte_idx["te"]]) #测试集 y

dataset_tr = matDataset(data_tr_list, labels_tr_tensor)  #训练集 X+y
dataset_te = matDataset(data_test_list, labels_te_tensor)   #测试集 X+y
print(labels_tr_tensor.shape)  # torch.Size([13469, 15])


# 创建数据加载器 分批次
train_loader = DataLoader(dataset_tr, batch_size=128, shuffle=True)   # 减小批次运算
test_loader = DataLoader(dataset_te, batch_size=256, shuffle=False)

# 检查数据加载器输出的形状
for batch_samples, batch_labels in train_loader:
    # 将样本堆叠到一个张量并增加一个通道维度
    batch_samples = torch.stack(batch_samples, dim=1).unsqueeze(1)
    print(f"Batch samples shape: {batch_samples.shape}")  # 应为 (128, 1, 12, 5000)
    print(f"Batch labels shape: {batch_labels.shape}")  # 应为 (128, 15)
    break

