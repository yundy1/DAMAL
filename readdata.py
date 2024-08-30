import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.model_selection import train_test_split


def load_ptbxl_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# def one_hot_5(y_test):
#     labels = np.zeros((len(y_test), 5))
#     for i in range(len(y_test.values)):
#         if len(y_test.values[i])==0:
#             continue
#         if 'NORM' in y_test.values[i]:
#             labels[i,4]=1
#         if 'IMI' in y_test.values[i]:
#             labels[i,0]=1
#         if 'AMI' in y_test.values[i]:
#             labels[i,1]=1
#         if 'LMI' in y_test.values[i]:
#             labels[i,2]=1
#         if 'other'in y_test.values[i]:
#             labels[i,3]=1
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

    Y = pd.read_csv('E:/search/myworks/my_new_ptbxl_database.csv', index_col='id')
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

    # 聚合 diagnostic_class 列的值并计算频次

    class_counts = Y['diagnostic_class'].value_counts()

    # 打印频次统计结果
    print(class_counts)

    # 将结果保存到 CSV 文件
    class_counts.to_csv('./class_counts.csv', index=False)
    # 存储 Y['diagnostic_class']
    Y['diagnostic_class'].to_csv('./diagnostic_class.csv', index=False)


    # 使用 原始数据集的折数去分数据########################################
    strat_fold_train = Y[Y.strat_fold != fold_num]

    X_train = load_ptbxl_data(strat_fold_train, sampling_rate, path)
    y_train = strat_fold_train.diagnostic_class

    strat_fold_test = Y[Y.strat_fold == fold_num]
    #  根据Y 匹配出X
    X_test = load_ptbxl_data(strat_fold_test, sampling_rate, path)
    y_test = strat_fold_test.diagnostic_class

    print(X_train.shape)
    print(y_train.shape)
    ##############################################################



    # # 随机分割数据集，将训练集和测试集按比例划分
    # train_choose, test_choose, y_train, y_test = train_test_split(Y, Y['diagnostic_class'], test_size=0.2, random_state=42)

    # # 加载训练集数据
    # X_train = load_ptbxl_data(train_choose, sampling_rate, path)
    # print(X_train.shape)
    # print(y_train.shape)
    # # 加载测试集数据
    # X_test = load_ptbxl_data(test_choose, sampling_rate, path)


    if (int(num_class) == 15):
        y_train = one_hot_15(y_train)
        y_test = one_hot_15(y_test)



    X_train = np.array(X_train)
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

    return re_format(X_train), y_train, re_format(X_test), y_test

