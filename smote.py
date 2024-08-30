import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MultiLabelBinarizer


def load_ptbxl_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def one_hot_15(y_test):
    labels = np.zeros((len(y_test), 15))
    for i in range(len(y_test)):
        if len(y_test[i]) == 0:
            continue
        if 'NORM' in y_test[i]:
            labels[i, 14] = 1
        if 'AMI' in y_test[i]:
            labels[i, 0] = 1
        if 'ASMI' in y_test[i]:
            labels[i, 1] = 1
        if 'ALMI' in y_test[i]:
            labels[i, 2] = 1
        if 'IMI' in y_test[i]:
            labels[i, 3] = 1
        if 'ILMI' in y_test[i]:
            labels[i, 4] = 1
        if 'IPLMI' in y_test[i]:
            labels[i, 5] = 1
        if 'IPMI' in y_test[i]:
            labels[i, 6] = 1
        if 'INJIN' in y_test[i]:
            labels[i, 7] = 1
        if 'INJIL' in y_test[i]:
            labels[i, 8] = 1
        if 'INJAS' in y_test[i]:
            labels[i, 9] = 1
        if 'INJAL' in y_test[i]:
            labels[i, 10] = 1
        if 'INJLA' in y_test[i]:
            labels[i, 11] = 1
        if 'LMI' in y_test[i]:
            labels[i, 12] = 1
        if 'PMI' in y_test[i]:
            labels[i, 13] = 1
    return labels


def read_ptbxl(num_class, fold_num):
    path = 'E:/search/myworks/yuanban/model/'
    sampling_rate = 500

    Y = pd.read_csv(
        'E:/search/myworks/new_ptbxl_database.csv', index_col='id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg_df = pd.read_csv('E:/search/myworks/fenlei.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                if int(num_class) == 2:
                    tmp.append(agg_df.loc[key].diagnostic_class)  # 选择不同的分类 2分类MI+NORM
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

    print(X_train.shape)
    print(y_train.shape)

    mlb = MultiLabelBinarizer()
    y_train_binary = mlb.fit_transform(y_train)

    # 展平X_train，使其形状为 (13726, 5000*12)
    nsamples, nx, ny = X_train.shape
    X_train_flat = X_train.reshape((nsamples, nx * ny))

    X_resampled_list = []
    y_resampled_list = []

    batch_size = 2000  # 设置批次大小，根据内存情况调整
    for start in range(0, X_train_flat.shape[0], batch_size):
        end = min(start + batch_size, X_train_flat.shape[0])
        X_batch = X_train_flat[start:end]
        y_batch = y_train_binary[start:end]

        for i in range(y_train_binary.shape[1]):
            adasyn = ADASYN(random_state=42)
            try:
                X_res_i, y_res_i = adasyn.fit_resample(X_batch, y_batch[:, i])
                if i == 0:
                    X_resampled_batch = X_res_i
                    y_resampled_batch = y_res_i.reshape(-1, 1)
                else:
                    X_resampled_batch = np.concatenate((X_resampled_batch, X_res_i), axis=0)
                    y_resampled_batch = np.concatenate((y_resampled_batch, y_res_i.reshape(-1, 1)), axis=0)
            except MemoryError:
                print(f"Memory error at batch {start}-{end}, label {i}")
                continue

        X_resampled_list.append(X_resampled_batch)
        y_resampled_list.append(y_resampled_batch)

    X_resampled = np.concatenate(X_resampled_list, axis=0)
    y_resampled = np.concatenate(y_resampled_list, axis=0)

    # 恢复X_resampled的形状
    nsamples_res, nfeatures_res = X_resampled.shape
    X_resampled = X_resampled.reshape((nsamples_res, nx, ny))

    y_resampled = mlb.inverse_transform(y_resampled)

    print(X_resampled.shape)
    print(len(y_resampled))

    if int(num_class) == 15:
        y_resampled = one_hot_15(y_resampled)
        y_test = one_hot_15(mlb.transform(y_test))

    X_resampled = np.array(X_resampled)
    X_test = np.array(X_test)

    def re_format(X):
        X = X.transpose(2, 0, 1).tolist()
        X_te = []
        for i, a in enumerate(X):
            c = []
            for j, b in enumerate(a):
                c.append(np.array(b))
            X_te.append(np.array(c))
        return X_te

    return re_format(X_resampled), y_resampled, re_format(X_test), y_test


data_tr_list, labels_tr, data_te_list, labels_te = read_ptbxl(15, 10)
