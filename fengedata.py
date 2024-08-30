import numpy as np
import torch

# from ptbxl.zqy.pre import data_train_list, data_val_list, data_test_list, labels_train, labels_val, labels_test
from torch.utils.data import TensorDataset, DataLoader
import ast
import numpy as np
import pandas as pd
import torch
import wfdb
# from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def load_ptbxl_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def one_hot_15(y_test):
    labels = np.zeros((len(y_test), 15))
    for i in range(len(y_test.values)):
        if len(y_test.values[i]) == 0:
            continue
        if 'NORM' in y_test.values[i]:
            labels[i, 14] = 1
        if 'AMI' in y_test.values[i]:
            labels[i, 0] = 1
        if 'ASMI' in y_test.values[i]:
            labels[i, 1] = 1
        if 'ALMI' in y_test.values[i]:
            labels[i, 2] = 1
        if 'IMI' in y_test.values[i]:
            labels[i, 3] = 1
        if 'ILMI' in y_test.values[i]:
            labels[i, 4] = 1
        if 'IPLMI' in y_test.values[i]:
            labels[i, 5] = 1
        if 'IPMI' in y_test.values[i]:
            labels[i, 6] = 1
        if 'INJIN' in y_test.values[i]:
            labels[i, 7] = 1
        if 'INJIL' in y_test.values[i]:
            labels[i, 8] = 1
        if 'INJAS' in y_test.values[i]:
            labels[i, 9] = 1
        if 'INJAL' in y_test.values[i]:
            labels[i, 10] = 1
        if 'INJLA' in y_test.values[i]:
            labels[i, 11] = 1
        if 'LMI' in y_test.values[i]:
            labels[i, 12] = 1
        if 'PMI' in y_test.values[i]:
            labels[i, 13] = 1
    return labels

def read_ptbxl(num_class, fold_num):
    path = 'E:/search/myworks/yuanban/model/'
    sampling_rate = 500

    Y = pd.read_csv('E:/search/myworks/new_ptbxl_database.csv', index_col='id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg_df = pd.read_csv('E:/search/myworks/fenlei.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                if int(num_class) == 2:
                    tmp.append(agg_df.loc[key].diagnostic_class) #选择不同的分类 2分类MI+NORM
                elif int(num_class) == 5:
                    tmp.append(agg_df.loc[key].diagnostic_subclass)
                elif int(num_class) == 7:
                    tmp.append(agg_df.loc[key].diagnostic_7class)
                elif int(num_class) == 15:
                    tmp.append(agg_df.loc[key].minclass)
        return list(set(tmp))

    Y['diagnostic_class'] = Y.scp_codes.apply(aggregate_diagnostic)
    print(Y.diagnostic_class.value_counts())

    strat_fold_train = Y[Y.strat_fold != fold_num]

    X_train = load_ptbxl_data(strat_fold_train, sampling_rate, path)
    y_train = strat_fold_train.diagnostic_class

    strat_fold_test = Y[Y.strat_fold == fold_num]
    X_test = load_ptbxl_data(strat_fold_test, sampling_rate, path)
    y_test = strat_fold_test.diagnostic_class



    if int(num_class) == 15:
        y_train = one_hot_15(y_train)
        y_test = one_hot_15(y_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(X_train.shape)
    print(y_train.shape)

    def re_format(X):
        X = X.transpose(2, 0, 1).tolist()
        X_te = []
        for i, a in enumerate(X):
            c = []
            for j, b in enumerate(a):
                c.append(np.array(b))
            X_te.append(np.array(c))
        return X_te

    return re_format(X_train), y_train, re_format(X_test), y_test

# 假设 read_ptbxl 函数已经定义
dataset = 'PTBXL'
fold_num = 2  # 改变折数,不影响随便
X_train, y_train, X_test, y_test = read_ptbxl(15, fold_num)

X_test = np.stack(X_test, axis=1) # 调整X维度 (1513, 12, 5000)
# X_test = X_test.transpose(1, 0, 2)  # 现在形状是 (1513, 12, 5000)
print(X_test.shape)
# 划分验证集和测试集，测试集占比 50%
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# 将 X_test 和 X_val 恢复为原始格式
X_test = [X_test[:, i, :] for i in range(X_test.shape[1])]
X_val = [X_val[:, i, :] for i in range(X_val.shape[1])]

def crop_resize(signal, crop_size, resize_size):
    """
    对信号进行裁剪和调整大小。
    :param signal: 输入信号，形状为 (13469,5000)
    :param crop_size: 裁剪的大小
    :param resize_size: 调整后的大小
    :return: 增强后的信号，形状为 (13469，resize_size)
    """
    if len(signal) < crop_size:
        # 如果信号长度不足，循环补充信号
        repeats = np.ceil(crop_size / len(signal)).astype(int)
        signal = np.tile(signal, repeats)[:crop_size]
    start = np.random.randint(0, len(signal) - crop_size + 1)
    cropped = signal[start:start + crop_size]
    resized = np.interp(np.linspace(0, crop_size, resize_size), np.arange(crop_size), cropped)
    return resized


def augment_data(X, y, crop_size, resize_size, augment_factor, minority_threshold=0.1):
    """
    对少数类进行数据增强。
    :param X: 输入数据，形状为 12个 (n_samples, 5000)
    :param y: 标签，形状为 (n_samples, n_classes)
    :param crop_size: 裁剪的大小
    :param resize_size: 调整后的大小
    :param augment_factor: 增强倍数
    :param minority_threshold: 少数类的阈值（相对于总样本数的比例）
    :return: 增强后的数据和标签
    """
    class_counts = np.sum(y, axis=0)
    minority_classes = np.where(class_counts < (minority_threshold * len(y)))[0]

    X_augmented = [[] for _ in range(len(X))]
    y_augmented = []

    for i in range(len(y)):
        for j in range(len(X)):
            X_augmented[j].append(X[j][i])
        y_augmented.append(y[i])
        if any(y[i, class_idx] == 1 for class_idx in minority_classes):
            for _ in range(augment_factor):
                for j in range(len(X)):
                    X_augmented[j].append(crop_resize(X[j][i], crop_size, resize_size))
                y_augmented.append(y[i])

    return [np.array(x) for x in X_augmented], np.array(y_augmented)


# 进行数据增强
crop_size = 4000  # 裁剪后的大小
resize_size = 5000  # 调整后的大小
augment_factor = 25  # 增强倍数 50内存不足？？？
minority_threshold = 0.1  # 少数类阈值
# 增强训练集！！！
print('X_train', X_train[0].shape)  # (13469, 5000)
print('y_train', y_train.shape)
X_train_augmented, y_train_augmented = augment_data(X_train, y_train, crop_size, resize_size, augment_factor, minority_threshold)  # 训练集倍增25
print('X_train_augmented', X_train_augmented[0].shape, len(X_train_augmented))   # X_train_augmented (51518, 5000) 12
print('y_train_augmented', y_train_augmented.shape)  # y_train_augmented (51518, 15)


# 不  增强测试集！！！
# print('X_test', X_test[0].shape)  # (1513, 5000)
# print('y_test', y_test[0].shape)
# X_test_augmented, y_test_augmented = augment_data(X_test, y_test, crop_size, resize_size, 25, minority_threshold)   # 测试集倍增15
# print('X_test_augmented', X_test_augmented[0].shape, len(X_test_augmented))   # X_test_augmented (5613, 5000) 12
# print('y_test_augmented', y_test_augmented.shape)  # y_test_augmented ((5613, 15)
#

########提取部分测试集里面的数据和原始测试集融合作为新的测试集
#
# # num_samples_to_move = 1800
# # # 随机选择索引
# # selected_indices = np.random.choice(len(X_train_augmented[0]), num_samples_to_move, replace=False)
#
# # 使用选定的样本创建新的测试集
# X_test_new = [lead[selected_indices] for lead in X_train_augmented]
# y_test_new = y_train_augmented[selected_indices]

# 从训练集中移除选定的样本
# X_train_augmented = [np.delete(lead, selected_indices, axis=0) for lead in X_train_augmented]
# y_train_augmented = np.delete(y_train_augmented, selected_indices, axis=0)

# 从训练集中逐步移除选定的样本，因为内存不够，按导联来逐个删除！！！
# for i, lead in enumerate(X_train_augmented):
#     X_train_augmented[i] = np.delete(lead, selected_indices, axis=0)
# y_train_augmented = np.delete(y_train_augmented, selected_indices, axis=0)

# # 将新的测试样本与现有的测试集 X_test_augmented 组合
# X_test_combined = [np.concatenate((X_test_augmented[i], X_test_new[i]), axis=0) for i in range(len(X_test_augmented))]
# y_test_combined = np.concatenate((y_test_augmented, y_test_new), axis=0)

# 打印形状以验证更改
print('X_train_augmented', X_train_augmented[0].shape, len(X_train_augmented))   # X_train_augmented shape
print('y_train_augmented', y_train_augmented.shape)  # y_train_augmented shape
print('X_test', X_test[0].shape, len(X_test))   # X_test_combined shape
print('y_test', y_test.shape)  # y_test_combined shape
print('X_val', X_val[0].shape, len(X_val))   # 验证集
print('y_val', y_val.shape)  # y_test_combined shape


# # 将增强后的数据转换为Tensor
X_train_augmented = [torch.tensor(x, dtype=torch.float32) for x in X_train_augmented]
y_train_augmented = torch.tensor(y_train_augmented, dtype=torch.float32)

X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]
y_test = torch.tensor(y_test, dtype=torch.float32)

X_val = [torch.tensor(x, dtype=torch.float32) for x in X_val]
y_val = torch.tensor(y_val, dtype=torch.float32)

# 创建数据加载器
class MultiChannelDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [x[idx] for x in self.X], self.y[idx]


train_dataset = MultiChannelDataset(X_train_augmented, y_train_augmented)
test_dataset = MultiChannelDataset(X_test, y_test)
val_dataset = MultiChannelDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
# 保存增强后的训练数据
np.savez('train_data1.npz',
         X_train_augmented_0=X_train_augmented[0],
         X_train_augmented_1=X_train_augmented[1],
         X_train_augmented_2=X_train_augmented[2],
         X_train_augmented_3=X_train_augmented[3],
         X_train_augmented_4=X_train_augmented[4],
         X_train_augmented_5=X_train_augmented[5],
         X_train_augmented_6=X_train_augmented[6],
         X_train_augmented_7=X_train_augmented[7],
         X_train_augmented_8=X_train_augmented[8],
         X_train_augmented_9=X_train_augmented[9],
         X_train_augmented_10=X_train_augmented[10],
         X_train_augmented_11=X_train_augmented[11],
         y_train_augmented=y_train_augmented.numpy())

# 保存测试数据
np.savez('test_data1.npz',
         X_test_0=X_test[0],
         X_test_1=X_test[1],
         X_test_2=X_test[2],
         X_test_3=X_test[3],
         X_test_4=X_test[4],
         X_test_5=X_test[5],
         X_test_6=X_test[6],
         X_test_7=X_test[7],
         X_test_8=X_test[8],
         X_test_9=X_test[9],
         X_test_10=X_test[10],
         X_test_11=X_test[11],
         y_test=y_test.numpy())
np.savez('val_data1.npz',
         X_val_0=X_val[0],
         X_val_1=X_val[1],
         X_val_2=X_val[2],
         X_val_3=X_val[3],
         X_val_4=X_val[4],
         X_val_5=X_val[5],
         X_val_6=X_val[6],
         X_val_7=X_val[7],
         X_val_8=X_val[8],
         X_val_9=X_val[9],
         X_val_10=X_val[10],
         X_val_11=X_val[11],
         y_val=y_val.numpy())
# 检查数据加载器输出的形状
for batch_samples, batch_labels in test_loader:
    print(f"Batch samples shape: {[x.shape for x in batch_samples]}")  # 应为 [(batch_size, 5000), ...]
    print(f"Batch labels shape: {batch_labels.shape}")  # 应为 (batch_size, 15)
    break
for batch_samples, batch_labels in val_loader:
    print(f"Batch samples shape: {[x.shape for x in batch_samples]}")  # 应为 [(batch_size, 5000), ...]
    print(f"Batch labels shape: {batch_labels.shape}")  # 应为 (batch_size, 15)
    break
