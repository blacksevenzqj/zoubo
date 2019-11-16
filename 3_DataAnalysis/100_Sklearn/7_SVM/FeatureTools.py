# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:48:40 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, cross_val_score as CVS
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import datetime
from time import time


# In[]:
# 路径设置
def set_file_path(path):
    import os
    os.chdir(path)


# 读入数据源
def readFile_inputData(train_name=None, test_name=None, index_col=None):
    if train_name is not None:
        train = pd.read_csv(train_name, index_col=index_col)
    if test_name is not None:
        test = pd.read_csv(test_name, index_col=index_col)
        return train, test
    else:
        return train


# 保存数据
def writeFile_outData(data, path):
    data.to_csv(path)


# 合并数据源（列向）
def consolidated_data_col(train_X, train_y):
    return pd.concat([train_X, train_y], axis=1)


# 分离数据源（列向）
def separate_data_col(df, y_name):
    train_X = df.drop(y_name, axis=1)
    train_y = pd.DataFrame(df[y_name])
    return train_X, train_y


# 合并数据源（行向）： 训练集 与 测试集 合并
def consolidated_data_row(train, test, y_name):
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_X, train_y = separate_data_col(train, y_name)
    all_data = pd.concat([train_X, test], axis=0).reset_index(drop=True)
    print("all_data size is : {}".format(all_data.shape))
    return all_data, train_y, ntrain, ntest


# 分离数据源（行向）： 训练集 与 测试集 分离
def separation_data_row(all_data, ntrain):
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    return train, test


# In[]:
# 恢复索引（删除数据后：如果X集恢复了索引，那么Y集也必须恢复索引）
def recovery_index(data_list):
    for i in data_list:
        i.index = range(i.shape[0])


# In[]:
# 缺失值（1:删特征; 2:删数据）
'''
笔记：“5.2、数据清洗”
5.2.2、缺失值处理：（对于 缺失值在80%甚至更高的情况 如下处理方式只能作为参考）
首选基于业务的填补方法，其次根据单变量分析进行填补，多重插补进行所有变量统一填补的方法只有在粗略清洗时才会使用。
1、缺失值少于20%
• 连续变量使用均值（正太分布）或 中位数（右偏）填补。
• 分类变量不需要填补，单算一类即可，或者用众数填补
2、缺失值在20%-80%
• 填补方法同上
• 另外每个有缺失值的变量生成一个指示哑变量，参与后续的建模（填补后的变量 和 缺失值指示变量 同时进模型，让模型来选择取舍）
3、缺失值在大于80%
• 每个有缺失值的变量生成一个指示哑变量，参与后续的建模，原始变量不使用。
'''


def missing_values_table(df, percent=None, del_type=1):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * mis_val / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    if percent is not None:
        if del_type == 1:
            temp_drop_col = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] > percent].index.tolist()
            df_nm = df.copy()
            df_nm.drop(temp_drop_col, axis=1, inplace=True)
            return mis_val_table_ren_columns, df_nm
        elif del_type == 2:
            temp_drop_col = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] > percent].index.tolist()
            df_nm = df.copy()
            for i in temp_drop_col:
                df_nm.drop(df_nm.loc[df_nm[i].isnull()].index, axis=0, inplace=True)
            return mis_val_table_ren_columns, df_nm
        else:
            return mis_val_table_ren_columns

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# 特征返回非缺失值部分
def get_notMissing_values(data_temp, feature):
    return data_temp[data_temp[feature] == data_temp[feature]]  # 返回全部Data


# 重复值处理
def duplicate_value(data, isDrop=False):
    # 重复项按特征统计
    print(data[data.duplicated()].count())
    # 去除重复项 后 长度
    nodup = data[-data.duplicated()]
    print("去除重复项后长度：%d" % len(nodup))
    # 去除重复项 后 长度
    print("去除重复项后长度：%d" % len(data.drop_duplicates()))
    # 重复项 长度
    print("重复项长度：%d" % (len(data) - len(nodup)))

    if isDrop:
        # 在原数据集上 删除重复项
        data.drop_duplicates(inplace=True)
        # 重设索引
        data = data.reset_index(drop=True)
        # data.index = range(data.shape[0])

    print(data.info())


# In[]:
# 缺失值填充
def missValue_all_fillna(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=[np.object]).columns
    # Median is my favoraite fillna mode, which can eliminate the skew impact.
    # 中位数是我最喜欢的fillna模式，它可以消除偏斜影响。
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # 分类（字符）用 字符"NA"填充
    df[cat_cols] = df[cat_cols].fillna("NA")


def missValue_fillna(df, feature_name, fill_val=None, val_type=1):
    if fill_val is not None:
        df[feature_name] = df[feature_name].fillna(fill_val)
    else:
        if val_type == 1:
            df[feature_name] = df[feature_name].fillna(df[feature_name].median())  # .median() 多特征为Series、单特征为值
        elif val_type == 2:
            df[feature_name] = df[feature_name].fillna(df[feature_name].mean())
        elif val_type == 3:
            df[feature_name] = df[feature_name].fillna(
                df[feature_name].mode().loc[0])  # .mode() 多特征为DataFrame、单特征为Series
        else:
            print("No Type You Choose")


# 缺失值填充： 分组填充（连续特征）
def missValue_group_fillna(df, nan_col, group_col, val_type=1):
    nan_index = df[nan_col][df[nan_col].isnull()].index

    #    # 方法一：
    #    gb_neigh_LF = df[nan_col].groupby(df[group_col])
    #    #gb_neigh_LF = df.groupby(df[group_col])[nan_col]
    #    for key,group in gb_neigh_LF:
    #        # 查找我们同时丢失值的位置和键存在的位置
    #        # 用键组对象的中位数填充这些空白
    #        lot_f_nulls_nei = df[nan_col].isnull() & (df[group_col] == key)
    #        if val_type == 1:
    #            temp_value = group.median()
    #        else:
    #            temp_value = group.mean()
    #        df.loc[lot_f_nulls_nei,nan_col] = temp_value

    # 方法二：
    if val_type == 1:
        temp_type = np.median  # 中位数（中位数 可以消除 偏斜 影响）
    elif val_type == 2:
        temp_type = np.mean  # 均值
    elif val_type == 3:
        temp_type = np.mode  # 众数

    df[nan_col] = df[nan_col].fillna(
        df.groupby(group_col)[nan_col].transform(temp_type))  # 或 .transform(lambda x: x.fillna(x.median()))

    nan_data = df.loc[nan_index, nan_col]

    return nan_data


# In[]:
# 离群值检测： 标准化（均值=0、标准差=1）后排序检测
def standardScaler_outlier(df, name):
    from sklearn.preprocessing import StandardScaler

    ss_val = StandardScaler().fit_transform(df[name][:, np.newaxis]);
    low_range = ss_val[ss_val[:, 0].argsort()][:10]
    high_range = ss_val[ss_val[:, 0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)


# 离群值检测： 使用 箱型图、散点趋势图 观测离群值
def outlier_detection(X, feature, y, y_name):
    ntrain = y[y_name].notnull().sum()
    X = X[0:ntrain]

    iqr = X[feature].quantile(0.75) - X[feature].quantile(0.25)

    # 利用 众数 减去 中位数 的差值  除以  四分位距来 查找是否有可能存在异常值
    temp_outlier = abs((X[feature].mode().iloc[0,] - X[feature].median()) / iqr)
    print(temp_outlier)

    # 如果值很大，需要进一步用直方图观测，对嫌疑大的变量进行可视化分析
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    con_data_distribution(X, feature, axes)
    # 从直方图中可以看出： 如果数据有最大峰值，属于正常数据，不用清洗。

    upper_point = X[feature].quantile(0.75) + 1.5 * iqr
    down_point = X[feature].quantile(0.25) - 1.5 * iqr

    upper_more_index = X[X[feature] > upper_point].index
    down_more_index = X[X[feature] < down_point].index

    #    print(X.iloc[upper_more_index].shape)
    #    print(y.iloc[upper_more_index].shape)

    f, axes = plt.subplots(1, 1, figsize=(23, 8))
    sns.regplot(X[feature], y[y_name], ax=axes)

    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    sns.regplot(X.loc[upper_more_index][feature], y.loc[upper_more_index][y_name], ax=axes[0])
    sns.regplot(X.loc[down_more_index][feature], y.loc[down_more_index][y_name], ax=axes[1])


# 删除离群值
def delete_outliers(X_Seriers, X_name, X_value, y_Seriers, y_name, y_value):
    # 多条件查询方式1：
    #    del_index = X_Seriers.loc[(X_Seriers[X_name]>X_value) & (y_Seriers[y_name]<y_value)].index
    # 多条件查询方式2：
    del_index = X_Seriers.loc[
        (X_Seriers[X_name].map(lambda x: x > X_value)) & (y_Seriers[y_name].map(lambda x: x < y_value))].index
    if id(X_Seriers) == id(y_Seriers):
        X_Seriers.drop(del_index, axis=0, inplace=True)
    else:
        X_Seriers.drop(del_index, axis=0, inplace=True)  # 如果X集恢复了索引，那么Y集也必须恢复索引
        y_Seriers.drop(del_index, axis=0, inplace=True)


# In[]:
# 分类模型 数据类别 样本不均衡（训练集 与 测试集）
def sample_category(ytest, ytrain):
    train_unique_label, train_counts_label = np.unique(ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(ytest, return_counts=True)
    print('-' * 60)
    print('Label Distributions: \n')
    print("训练集类别%s，数量%s，占比%s" % (train_unique_label, train_counts_label, (train_counts_label / len(ytrain))))
    print("测试集类别%s，数量%s，占比%s" % (test_unique_label, test_counts_label, (test_counts_label / len(ytest))))


# 分类模型 数据类别 样本不均衡（单一数据集测试）
def Sample_imbalance(data, y_name):
    print(data.shape)
    print(data.info())

    print('Y = 0', round(len(data[data[y_name] == 0]) / len(data) * 100, 2), "% of the dataset")
    print('Y = 1', round(len(data[data[y_name] == 1]) / len(data) * 100, 2), "% of the dataset")

    # 查看目标列Y的情况
    print(data.groupby(y_name).size())
    print(data[y_name].value_counts())

    # 目标变量Y分布可视化
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    sns.countplot(x=y_name, data=data, ax=axs[0])  # 柱状图
    axs[0].set_title("Frequency of each TARGET")  # 每个目标的出现频率

    data[y_name].value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')  # 饼图
    axs[1].set_title("Percentage of each TARGET")  # 每个目标的百分比
    plt.show()


# 集合 交、差、补
def set_diff(set_one, set_two):
    temp_list = []
    temp_list.append(list(set(set_one) & set(set_two)))  # 交
    temp_list.append(list(set(set_one) - (set(set_two))))  # 差
    temp_list.append(list(set(set_one) ^ set(set_two)))  # 补
    return temp_list


# 所有的 分类类别 都使用str，而不使用 categorical类型（有的库抱异常）
# 设置分类变量类型：
def set_classif_col(df, feature_name, val_type=1):
    if val_type == 1:
        temp_type = int
    elif val_type == 2:
        temp_type = float
    elif val_type == 3:
        temp_type = str
    else:
        raise Exception('Val Type is Error')
    df[feature_name] = df[feature_name].astype(temp_type)


# 设置分类变量类型为：category（很不用）
def set_col_category(df, feature_name, categories_=None):
    if categories_ is None:
        df[feature_name] = df[feature_name].astype('category', ordered=True)  # 自动排序： 按首字母顺序
    else:
        df[feature_name] = df[feature_name].astype('category', ordered=True, categories=categories_)  # 手动排序


# 分类特征 普通编码：
def classif_labelEncoder(data, cols):
    from sklearn.preprocessing import LabelEncoder

    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))


# In[]:
# 数据切分
def data_segmentation_skf(X, y, test_size=0.3):
    n_splits_temp = int(1 / test_size)
    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    # StratifiedKFold 其实是 5折 交叉验证 的 分层采样： 这里用于 将原始数据集 分为 训练集 和 测试集（共5次循环，其实一次就够了）
    sss = StratifiedKFold(n_splits=n_splits_temp, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):  # 每一次循环赋值 都 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
        print("Train:", train_index, "Train_len:", len(train_index), "Test:", test_index, "Test_len:", len(test_index))
        print(len(train_index) / len(y), len(test_index) / len(y))
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
        break

    return original_Xtrain, original_Xtest, original_ytrain, original_ytest


# In[]:
# ================================数据分布==============================
# In[]:
# 分类模型 连续特征 数据分布： （不能有缺失值）
def class_data_distribution(data, feature, label, axes):
    data = get_notMissing_values(data, feature)
    sns.distplot(data[feature], bins=100, color='green', ax=axes[0][0])
    axes[0][0].set_title('feature: ' + str(feature))
    axes[0][0].set_xlabel('')

    sns.boxplot(y=feature, data=data, ax=axes[0][1])
    axes[0][1].set_title('feature: ' + str(feature))
    axes[0][1].set_ylabel('')

    sns.distplot(data[feature][data[label] == 1], bins=100, color='red', ax=axes[1][0])
    sns.distplot(data[feature][data[label] == 0], bins=100, color='blue', ax=axes[1][0])
    axes[1][0].set_title('feature: ' + str(feature))
    axes[1][0].set_xlabel('')

    sns.boxplot(x=label, y=feature, data=data, ax=axes[1][1])
    axes[1][1].set_title('feature: ' + str(feature))
    axes[1][1].set_xlabel('')
    axes[1][1].set_ylabel('')


# 分类模型 连续特征 2个特征之间 散点分布：
def class_data_scatter(x_data, one_f, two_f, y, axes):
    axes.scatter(x_data[:, one_f], x_data[:, two_f], c=y, s=10, cmap="rainbow")  # 蓝色
    axes.set_xlabel(one_f)  # x轴标签
    axes.set_ylabel(two_f)  # y轴标签


# 连续/分类模型 连续 特征/因变量 直方图分布： （不能有缺失值）
'''
笔记：“5.2、数据清洗”
5.2.3.1、单变量离群值的发现：
5.2.3.1.1、极端值（考虑删除）
• 设置标准，如： 5倍标准差之外的数据
• 极值有时意味着错误，应重新理解数据，例如：特殊用户的超大额消费

5.2.3.1.2、离群值
5.2.3.1.2.1、平均值法：平均值±n倍标准差之外的数据
学生化残差  =  (残差 - 残差均值) / 残差的标准差
样本量为几百个时：|SR| > 2 为强影响点
样本量为上千个时：|SR| > 3 为强影响点
建议的临界值：
• |SR| > 2，用于观察值较少的数据集（离群值 > 平均值+2倍标准差）
• |SR| > 3，用于观察值较多的数据集（离群值 > 平均值+3倍标准差）
5.2.3.1.2.2、四分位数法：
• IQR = Q3 – Q1
• 下区间：Q1 – 1.5 x IQR，上区间：Q3 + 1.5 x IQR   更适用于对称分布的数据

5.2.3.1.3、离群值的处理：
5.2.3.1.3.1、盖帽法：
5.2.3.1.3.2、分箱法：
'''


def con_data_distribution(data, feature, axes):
    data = get_notMissing_values(data, feature)

    sns.set()  # 切换到seaborn的默认运行配置
    sns.distplot(data[feature], bins=100, fit=scipy.stats.norm, color='green', ax=axes[0])  # rug=True分布观测条显示
    # Get the fitted parameters used by the function
    (mu, sigma) = scipy.stats.norm.fit(data[feature])  # mu均值、sigma标准差： 也就是图中黑色线所示标准正太分布
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # mu - sigma  →  mu + sigma = 68%
    axes[0].plot((mu - sigma, mu - sigma), (0, 1), c='r', lw=1.5, ls='--', alpha=0.3)  # 68%
    axes[0].plot((mu + sigma, mu + sigma), (0, 1), c='r', lw=1.5, ls='--', alpha=0.3)

    # mu - 2sigma  →  mu + 2sigma = 95%
    axes[0].plot((mu - 2 * sigma, mu - 2 * sigma), (0, 1), c='black', lw=1.5, ls='--', alpha=0.3)  # 95%
    axes[0].plot((mu + 2 * sigma, mu + 2 * sigma), (0, 1), c='black', lw=1.5, ls='--', alpha=0.3)

    # mu - 3sigma  →  mu + 3sigma = 99%
    axes[0].plot((mu - 3 * sigma, mu - 3 * sigma), (0, 1), c='b', lw=1.5, ls='--', alpha=0.3)  # 99%
    axes[0].plot((mu + 3 * sigma, mu + 3 * sigma), (0, 1), c='b', lw=1.5, ls='--', alpha=0.3)

    # mu - 5sigma  →  mu + 5sigma = 5倍标准差之外的数据
    if len(data.loc[data[feature] < mu - 5 * sigma, feature].index) > 0:
        axes[0].plot((mu - 5 * sigma, mu - 5 * sigma), (0, 1), c='y', lw=1.5, ls='--', alpha=0.3)  # 5倍标准差之外的数据
    if len(data.loc[data[feature] > mu + 5 * sigma, feature].index) > 0:
        axes[0].plot((mu + 5 * sigma, mu + 5 * sigma), (0, 1), c='y', lw=1.5, ls='--', alpha=0.3)
    # Now plot the distribution
    axes[0].legend(['Normal dist. ( $\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
    axes[0].set_title('feature: ' + str(feature))
    axes[0].set_xlabel('')

    sns.boxplot(y=feature, data=data, ax=axes[1])
    axes[1].set_title('feature: ' + str(feature))
    axes[1].set_ylabel('')


# 连续模型 连续特征 与 Y 散点分布：
'''
方差齐次性： 测试两个特征的均方差的最佳方法是图形方式。 
通过圆锥（在图形的一侧较小的色散，在相反侧的较大色散）或菱形（在分布中心的大量点）来表示偏离均等色散的形状。
'''


def con_data_scatter(x_data, i, y, j):
    f, axes = plt.subplots(2, 1, figsize=(15, 15))

    axes[0].scatter(x_data[i], y[j], c='#0000FF', s=10, cmap="rainbow")  # 蓝色
    axes[0].set_xlabel(i)  # x轴标签
    axes[0].set_ylabel(j)  # y轴标签

    sns.regplot(x_data[i], y[j], ax=axes[1])


# 连续模型 分类特征 与 连续因变量Y 四分位图：
# 分类模型 连续特征 与 分类因变量Y 四分位图：
# 可以作为 斯皮尔曼相关系数 辅助可视化分析： 分类特征 对 连续因变量Y 有用； 连续特征 对 分类因变量Y 有用。
def box_diagram(data, x_axis_name, y_axis_name, axes, ymin=None, ymax=None):
    if ymin is not None and ymax is not None:
        axes.axis(ymin=ymin, ymax=ymax)
    sns.boxplot(x=x_axis_name, y=y_axis_name, data=data, ax=axes)


# In[]:
# -----------------------------正太、偏度检测-------------------------------
# In[]:
# 正太分布检测：
'''
原假设：样本来自一个正态分布的总体。
备选假设：样本不来自一个正态分布的总体。
w和p同向： w值越小； p-值越小、接近于0； 拒绝原假设。
'''


def normal_distribution_test(data):
    var = data.columns
    shapiro_var = {}
    for i in var:
        shapiro_var[i] = scipy.stats.shapiro(data[i])  # 返回 w值 和 p值

    shapiro = pd.DataFrame(shapiro_var).T.sort_values(by=1, ascending=False)

    fig, axe = plt.subplots(1, 1, figsize=(15, 10))
    axe.bar(shapiro.index, shapiro[0], width=.4)  # 自动按X轴---skew.index索引0-30的顺序排列
    axe.set_title("Normal distribution for shapiro")

    # 在柱状图上添加数字标签
    for a, b in zip(shapiro.index, shapiro[0]):
        # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)
    plt.show()


'''
偏度>1 为偏斜数据，需要取log
Skewness:偏度是描述数据分布形态的统计量，其描述的是某总体取值分布的对称性，简单来说就是数据的不对称程度。
偏度是三阶中心距计算出来的。
（1）Skewness = 0 ，分布形态与正态分布偏度相同。
（2）Skewness > 0 ，正偏差数值较大，为正偏或右偏。长尾巴拖在右边，数据右端有较多的极端值。
（3）Skewness < 0 ，负偏差数值较大，为负偏或左偏。长尾巴拖在左边，数据左端有较多的极端值。
（4）数值的绝对值越大，表明数据分布越不对称，偏斜程度大。
计算公式：
Skewness=E[((x-E(x))/(\sqrt{D(x)}))^3]
'''


def skew_distribution_test(data):
    #    var = data.columns
    #    skew_var = {}
    #    for i in var:
    #        skew_var[i] = abs(data[i].skew())
    #    skew = pd.Series(skew_var).sort_values(ascending=False)

    skew = np.abs(data.skew()).sort_values(ascending=False)
    # 下面这种计算方式 在无缺失值情况下 和 上述2种计算方式 有小数点后两位的 差异。 有缺失值情况 待进一步验证。
    #    skew = data.apply(lambda x: np.abs(skew(x.dropna()))).sort_values(ascending=False)

    kurt = np.abs(data.kurt()).sort_values(ascending=False)

    fig, axe = plt.subplots(2, 1, figsize=(18, 20))

    # bar的X轴顺序问题： https://blog.csdn.net/qq_35318838/article/details/80198307
    axe[0].bar(np.arange(len(skew.index)), skew, width=.4)
    axe[0].set_xticks(np.arange(len(skew.index)))
    axe[0].set_xticklabels(skew.index)
    axe[0].set_title("Normal distribution for skew")
    # 在柱状图上添加数字标签
    for a, b in zip(np.arange(len(skew.index)), skew):
        # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
        axe[0].text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)

    axe[1].bar(np.arange(len(kurt.index)), kurt, width=.4)
    axe[1].set_xticks(np.arange(len(kurt.index)))
    axe[1].set_xticklabels(kurt.index)
    axe[1].set_title("Normal distribution for kurt")
    # 在柱状图上添加数字标签
    for a, b in zip(np.arange(len(kurt.index)), kurt):
        # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
        axe[1].text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)

    plt.show()

    return skew, kurt


'''
峰度
Kurtosis:峰度是描述某变量所有取值分布形态陡缓程度的统计量，简单来说就是数据分布顶的尖锐程度。
峰度是四阶标准矩计算出来的。
（1）Kurtosis=0 与正态分布的陡缓程度相同。
（2）Kurtosis>0 比正态分布的高峰更加陡峭——尖顶峰
（3）Kurtosis<0 比正态分布的高峰来得平台——平顶峰
计算公式：
Kurtosis=E[((x-E(x))/ (\sqrt(D(x))))^4]-3
'''


def normal_comprehensive(data, skew_limit=1):  # skew_limit=0.75
    if type(data) == pd.core.series.Series:
        temp_data = pd.DataFrame(data.copy())
    else:
        temp_data = data.copy()

    normal_distribution_test(temp_data)
    skew, kurt = skew_distribution_test(temp_data)

    var_x_ln = skew.index[skew > skew_limit]  # skew的索引 --- data的列名
    print(var_x_ln, len(var_x_ln))
    for i, var in enumerate(var_x_ln):
        f, axes = plt.subplots(1, 2, figsize=(23, 8))
        con_data_distribution(temp_data, var, axes)

    # 将偏度大于1的连续变量 取对数
    if len(var_x_ln) > 0:
        logarithm_nagative(temp_data, var_x_ln, 2)

        normal_distribution_test(temp_data)
        skew, kurt = skew_distribution_test(temp_data)

        var_x_ln = skew.index[skew > skew_limit]  # skew的索引 --- data的列名
        print(var_x_ln, len(var_x_ln))
        for i, var in enumerate(var_x_ln):
            f, axes = plt.subplots(1, 2, figsize=(23, 8))
            con_data_distribution(temp_data, var, axes)


# Q-Q图检测
def normal_QQ_test(data, feature):
    temp_data = data.copy()

    # statsmodels包实现
    import statsmodels.api as sm
    fig = sm.qqplot(temp_data[feature], fit=True, line='45')
    fig.show()

    # scipy包实现
    fig = plt.figure()
    scipy.stats.probplot(temp_data[feature], plot=plt)
    fig.show()

    # ========================================================================

    # 手动
    mean = temp_data[feature].mean()
    std = temp_data[feature].std()
    temp_data.sort_values(by=feature, inplace=True)

    # 注意： Data.列名 是传统调用方式，但 Data.index 是特殊的，是DataFrame的索引，不是列名。
    s_r = temp_data.reset_index(drop=False)
    s_r['p'] = (s_r.index - 0.5) / len(s_r)  # 计算百分位数 p(i)
    # print(s_r['p'])
    s_r['q'] = (s_r[feature] - mean) / std  # 计算q值：Z分数

    st = temp_data[feature].describe()
    x1, y1 = 0.25, st['25%']
    x2, y2 = 0.75, st['75%']

    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(3, 1, 1)  # 创建子图1
    # 绘制数据分布图
    ax1.scatter(temp_data.index, temp_data[feature])
    plt.grid()

    ax2 = fig.add_subplot(3, 1, 2)  # 创建子图2
    # 绘制直方图
    temp_data[feature].hist(bins=30, alpha=0.5, normed=True, ax=ax2)
    temp_data[feature].plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid()

    ax3 = fig.add_subplot(3, 1, 3)  # 创建子图3
    ax3.plot(s_r['p'], s_r[feature], 'k.', alpha=0.1)
    ax3.plot([x1, x2], [y1, y2], '-r')  # 绘制QQ图，直线为 四分之一位数、四分之三位数的连线，基本符合正态分布
    plt.grid()
    fig.show()


# In[]:
# -----------------------------正太、偏度检测-------------------------------
# In[]:
# ================================数据分布==============================


# In[]:
# 取对数（可处理负数）
def logarithm(X_rep, var_x_ln, f_type=1):
    if f_type == 1:
        for i in var_x_ln:
            if min(X_rep[i]) <= 0:
                # 只要最小值<=0： 整体数据 向右 平移 最小值的绝对值 + 0.01。 最后再取log。
                X_rep[i + "_ln"] = np.log(X_rep[i] + abs(min(X_rep[i])) + 0.01)  # 负数取对数的技巧
            else:
                X_rep[i + "_ln"] = np.log(X_rep[i])
    else:
        for i in var_x_ln:
            if min(X_rep[i]) <= 0:
                X_rep[i] = np.log(X_rep[i] + abs(min(X_rep[i])) + 0.01)  # 负数取对数的技巧
            else:
                X_rep[i] = np.log(X_rep[i])
    return X_rep


def logarithm_nagative(data, var_ln, f_type=1):
    rep = data.copy()
    for i in var_ln:
        if min(rep[i]) <= 0:
            if f_type == 1:
                rep["Less0_lable_" + i + "_ln"] = 0
                rep.loc[rep[i] <= 0, "Less0_lable_" + i + "_ln"] = 1
                rep["Less0_Absmin_" + i + "_ln"] = abs(min(rep[i]))
            else:
                rep["Less0_lable_" + i] = 0
                rep.loc[rep[i] <= 0, "Less0_lable_" + i] = 1
                rep["Less0_Absmin_" + i] = abs(min(rep[i]))
    return logarithm(rep, var_ln, f_type)


def re_logarithm_nagative(rep, var_ln, f_type=1):
    for i in var_ln:
        if f_type == 1:
            temp_col_name = "Less0_Absmin_" + i + "_ln"
            temp_i = i + "_ln"
        else:
            temp_col_name = "Less0_Absmin_" + i
            temp_i = i
        try:
            rep[temp_col_name]  # 如果 没有字段 抛异常，执行except的代码： 说明 取对数前 数据都是>=0的
            rep[temp_i] = np.exp(rep[temp_i]) - 0.01 - rep[temp_col_name]
        except:
            rep[temp_i] = np.exp(rep[temp_i])


# 取对数（不处理负数： 负数取对数后为np.nan）
def logarithm_log1p(rep, var_ln, f_type=1):
    if f_type == 1:
        for i in var_ln:
            rep[i + "_ln"] = np.log1p(rep[i])
    else:
        for i in var_ln:
            rep[i] = np.log1p(rep[i])


def re_logarithm_expm1(rep, var_ln, f_type=1):
    for i in var_ln:
        if f_type == 1:
            temp_i = i + "_ln"
        else:
            temp_i = i
        rep[temp_i] = np.expm1(rep[temp_i])


# 测试np.log 和 log1p 的区别：
def test_log_log1p(x=10 ** -16):
    print(x)
    print(np.log(x))
    print(np.exp(np.log(x)))

    print(np.log(x + 1))
    print(np.exp(np.log(x + 1)))

    print(np.log1p(x))
    print(np.expm1(np.log1p(x)))


# （高度）偏斜特征的 Box-Cox 变换
'''
http://onlinestatbook.com/2/transformations/box-cox.html
当 lanmuda = 0 时， boxcox1p 与 np.log1p 结果相同
当 lanmuda != 0 时， boxcox1p 如何变化的 还需研究
'''


def scipy_boxcox1p(df, skewed_features, lanmuda=0.15):
    from scipy.special import boxcox1p
    df[skewed_features] = boxcox1p(df[skewed_features], lanmuda)


# In[]:
# 相似度计算1
# 皮尔森相似度
'''
1、皮尔森相似度（带 因变量Y） 先选出对Y 有贡献的： corrFunction_withY
2、再看 特征共线性： corrFunction
3、最后再对 选出的特征 与 Y 做pairplot图： feature_scatterplotWith_y
'''


# 1、特征选择：（带 因变量Y）
def corrFunction_withY(data_corr, label, image_width=20, image_hight=18):  # label： 因变量Y名称
    corr = data_corr.corr()  # 计算各变量的相关性系数
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                           inplace=True)
    print(df_all_corr_abs[(df_all_corr_abs["Feature_1"] != label) & (df_all_corr_abs['Feature_2'] == label)])
    print()
    # 皮尔森相似度 排序
    df_all_corr = corr.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                       inplace=True)
    print(df_all_corr[(df_all_corr["Feature_1"] != label) & (df_all_corr['Feature_2'] == label)])

    temp_x = []
    for i, fe in enumerate(list(corr.index)):  # 也可以是：data_corr.columns
        temp_x.append("x" + str(i))
    xticks = temp_x  # x轴标签
    yticks = list(corr.index)  # y轴标签
    fig = plt.figure(figsize=(image_width, image_hight))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})  # 绘制相关性系数热力图
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
    plt.show()


# 2、特征选择：特征共线性（还可以做 方差膨胀系数）
def corrFunction(data_corr, image_width=20, image_hight=18):
    '''
    1、特征间共线性：两个或多个特征包含了相似的信息，期间存在强烈的相关关系
    2、常用判断标准：两个或两个以上的特征间的相关性系数高于0.8
    3、共线性的影响：
    3.1、降低运算效率
    3.2、降低一些模型的稳定性
    3.3、弱化一些模型的预测能力
    '''
    # 建立共线性表格（是检测特征共线性的，所以排除Y）
    correlation_table = data_corr.corr()
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = correlation_table.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                           inplace=True)
    temp_corr_abs = df_all_corr_abs[(df_all_corr_abs["Feature_1"] != df_all_corr_abs["Feature_2"])][::2]
    #    temp_corr_abs.to_csv(r"C:\Users\dell\Desktop\123123\temp_corr_abs.csv")
    print(temp_corr_abs)
    print()
    # 皮尔森相似度 排序
    df_all_corr = correlation_table.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                       inplace=True)
    temp_corr = df_all_corr[(df_all_corr["Feature_1"] != df_all_corr["Feature_2"])][::2]
    #    temp_corr.to_csv(r"C:\Users\dell\Desktop\123123\temp_corr.csv")
    print(temp_corr)

    # 热力图
    temp_x = []
    for i, fe in enumerate(data_corr.columns):
        temp_x.append("x" + str(i))
    xticks = temp_x  # x轴标签
    yticks = list(correlation_table.index)  # y轴标签
    fig = plt.figure(figsize=(image_width, image_hight))
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(correlation_table, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})  #
    ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=14)


# 3、最后再对 选出的特征 与 Y 做pairplot图
def feature_scatterplotWith_y(data, cols):
    sns.set()
    sns.pairplot(data[cols], size=2.5)  # scatterplot
    plt.show();


# 特征选择：（特征 与 因变量Y） 皮尔森
def feature_corrWith_y(X, y_series, top_num=20):
    if type(X) is pd.core.series.Series:
        X = pd.DataFrame(X)
    return np.abs(X.corrwith(y_series)).sort_values(ascending=False)[:top_num]


# 相似度计算2
# 斯皮尔曼相似度：
# 特征选择：（特征 与 因变量Y） 斯皮尔曼
def feature_spearmanrWith_y(X_series, y_series):
    r, p = scipy.stats.spearmanr(X_series, y_series)
    return r, p


# In[]:
# ================================线性回归特征分析==============================
# In[]:
from sklearn.metrics import mean_squared_error as MSE  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

# 拟合优度
# R^2
'''
从 真实值的内部差异（方差）为出发点思考 R^2 公式：
1、mse 代表 回归值与真实值之间的（均方误差MSE）  
mse = np.sum(np.square(y_predict - y_true))  /  len(y_ture)

2、variance 表示 真实值 的 内部差异（方差）
variance = np.sum(np.square(y_true - np.mean(y_true)))  /  len(y_ture)

3、R^2
R^2  =  1 - (mse/variance)  =  1 - (RSS残差平方和 / TSS总离差平方和)
注意： mse/variance 相除之后，各自分母的 len(y_ture) 被消去，得到上述公式：分子上得到RSS残差平方和，分母上这时不能说是方差（len(y_ture)被消除了），而是TSS总离差平方和。

4、所以，消去分母 len(y_ture) 后，公式变换得到 最终 R^2 公式：
rss = np.sum(np.square(y_predict - y_true))
tss = np.sum(np.square(y_true - np.mean(y_true)))
R^2 = 1 - (rss/tss)

因此R-squared既考量了回归值与真实值的差异，也兼顾了真实值的离散程度。【模型对样本数据的拟合度】R-squared 取值范围 (-∞,1]，
值越大表示模型越拟合训练数据，最优解是1；当模型预测为随机值的时候，R^2有可能为负；若预测值恒为样本期望，R^2为0。

其中y是我们的真实标签，y^^ 是我们的预测结果， y^- 是我们的均值， 分母 ∑(y - y^-)^2  如果除以样本量m就是我们的方差。
方差的本质是任意一个y值和样本均值的差异，差异越大，这些值所带的信息越多。在R^2中，分子（RSS离差平方和）是模型没有捕获到的信息总量，
分母是真实标签所带的信息量（可以理解为方差：m被消除了），所以其衡量的是：
1 - 模型没有捕获到的信息量 占 真实标签中所带的信息量的比例  =  模型捕获到的信息量 占 真实标签中所带的信息量的比例，
所以，R^2 越接近1越好。

上述公式 R^2 = 1 - RSS/TSS 是正确公式；而 R^2 = ESS/TSS 是不完整、有缺陷的公式（不能使用）
'''


def r2_score_customize(y_true, y_predict, customize_type=1):
    if customize_type == 1:
        return r2_score(y_true, y_predict)
    else:
        rss = np.sum(np.square(y_predict - y_true))
        tss = np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - (rss / tss)


# 如果解释不同解释变量的模型之间的模型优劣（样本一样）
# Adj.R^2
'''
i=1 当有截距项时，反之等于0
n=用于拟合该模型的观察值数量
p=模型中参数的个数
Adj.R^2  =  1  -  (n-i)(1-R^2) / (n-p) 
各处查询后用下面的公式：
Adj.R^2  =  1  -  (n-i)(1-R^2) / (n-p-1) 
'''


def adj_r2_customize(y_true, y_predict, coef_num, customize_type=1):
    r2_score = r2_score_customize(y_true, y_predict, customize_type)
    adj_r2 = 1 - ((len(y_true) - 1) * (1 - r2_score)) / (len(y_true) - coef_num - 1)
    return adj_r2


# 多元线性回归选择自变量指标（AIC/BIC、R^2/Adj.R^2）
# AIC： （有问题，公式都不能确定。。。）
'''
AIC指标： 多元线性回归选择自变量指标（AIC/BIC、R^2/Adj.R^2）
k = 解释变量个数
RSS = 残差的离差平方和
AIC = 2k + n(log(RSS/n)) 
'''
# def aic_customize(y_true, y_predict, coef_num):
#    rss = np.sum(np.square(y_predict - np.mean(y_true)))
#    aic = 2*coef_num + len(y_true)*(np.log(rss/len(y_true)))
#    return aic


# 残差分析： 误差项 即 扰动项 即 残差
'''
残差分析： 
残差中是否有离群值？
残差散点图是否和某个解释变量X有曲线关系？
残差的离散程度是否和某个解释变量有关？
残差 = (Ytrue - Yhat)
'''


# 1.1、扰动项ε 独立同分布 （异方差检验、DW检验）
# 1.1.1、异方差： 随着x的增大，残差呈扇面型分布，残差的方差呈放大趋势。出现在“横截面”数据中（样本是同一时间采集到的）
def heteroscedastic(X, Y, col_list):
    from statsmodels.formula.api import ols

    temp_X = X[col_list]
    temp_Y = Y
    temp_data = pd.concat([temp_X, temp_Y], axis=1)

    formula = "Y" + '~' + '+'.join(col_list)
    print(formula)
    lm_s1 = ols(formula, data=temp_data).fit()
    print(lm_s1.rsquared, lm_s1.aic)
    temp_data['Pred'] = lm_s1.predict(temp_data)
    temp_data['resid'] = lm_s1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s1.summary())

    print("-" * 30)

    logarithm(temp_data, "Y")
    formula = "Y_ln" + '~' + '+'.join(col_list)
    print(formula)
    lm_s2 = ols(formula, data=temp_data).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data['Pred'] = lm_s2.predict(temp_data)
    temp_data['resid'] = lm_s2.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s2.summary())

    print("-" * 30)

    col_list_ln = [i + "_ln" for i in col_list]
    logarithm(temp_data, col_list)
    formula = "Y_ln" + '~' + '+'.join(col_list_ln)
    print(formula)
    lm_s3 = ols(formula, data=temp_data).fit()
    print(lm_s3.rsquared, lm_s3.aic)
    temp_data['Pred'] = lm_s3.predict(temp_data)
    temp_data['resid'] = lm_s3.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s3.summary())

    r_sq = {'Y~Features': lm_s1.rsquared, 'ln(Y)~Features': lm_s2.rsquared, 'ln(Y)~ln(Features)': lm_s3.rsquared}
    return r_sq


def heteroscedastic_singe(X, Y, col):
    from statsmodels.formula.api import ols

    temp_X = X[col]
    temp_Y = pd.DataFrame(Y, columns=['Y'])
    temp_data = pd.concat([temp_X, temp_Y], axis=1)

    formula = "Y" + '~' + '+' + col
    print(formula)
    lm_s1 = ols(formula, data=temp_data).fit()
    print(lm_s1.rsquared, lm_s1.aic)
    temp_data['Pred'] = lm_s1.predict(temp_data)
    temp_data['resid'] = lm_s1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot(col, 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s1.summary())

    print("-" * 30)

    temp_data["Y_ln"] = np.log(temp_data["Y"])
    formula = "Y_ln" + '~' + '+' + col
    print(formula)
    lm_s2 = ols(formula, data=temp_data).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data['Pred'] = lm_s2.predict(temp_data)
    temp_data['resid'] = lm_s2.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot(col, 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s2.summary())

    print("-" * 30)

    temp_data[col + "_ln"] = np.log(temp_data[col])
    formula = "Y_ln" + '~' + '+' + col + "_ln"
    print(formula)
    lm_s3 = ols(formula, data=temp_data).fit()
    print(lm_s3.rsquared, lm_s3.aic)
    temp_data['Pred'] = lm_s3.predict(temp_data)
    temp_data['resid'] = lm_s3.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot(col + "_ln", 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s3.summary())

    r_sq = {'Y~' + col: lm_s1.rsquared, 'ln(Y)~' + col: lm_s2.rsquared, 'ln(Y)~ln(' + col + ')': lm_s3.rsquared}
    return r_sq


# 1.2、扰动项ε 服从正太分布 （QQ检验）
def disturbance_term_normal(X, Y, col_list):
    from statsmodels.formula.api import ols

    temp_X = X[col_list]
    temp_Y = Y
    temp_data = pd.concat([temp_X, temp_Y], axis=1)

    formula = "Y" + '~' + '+'.join(col_list)
    print(formula)
    lm_s1 = ols(formula, data=temp_data).fit()
    print(lm_s1.rsquared, lm_s1.aic)
    temp_data['Pred'] = lm_s1.predict(temp_data)
    temp_data['resid'] = lm_s1.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状
    print(lm_s1.summary())

    # 扰动项ε QQ检测
    normal_QQ_test(temp_data, "resid")


# 2.1、学生化残差： 强影响点分析
'''
学生化残差  =  (残差 - 残差均值) / 残差的标准差
样本量为几百个时：|SR| > 2 为强影响点
样本量为上千个时：|SR| > 3 为强影响点
'''


def studentized_residual(Xtrain, Ytrain):
    from statsmodels.formula.api import ols

    temp_Y = Ytrain
    temp_data = pd.concat([Xtrain, temp_Y], axis=1)
    cols = list(temp_data.columns)
    cols.remove("Y")
    cols_noti = cols
    formula = "Y" + '~' + '+'.join(cols_noti)

    lm_s = ols(formula, data=temp_data).fit()
    print(lm_s.rsquared, lm_s.aic)
    temp_data['Pred'] = lm_s.predict(temp_data)
    temp_data['resid'] = lm_s.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状

    temp_data['resid_t'] = (temp_data['resid'] - temp_data['resid'].mean()) / temp_data['resid'].std()

    temp_data2 = temp_data[abs(temp_data['resid_t']) <= 3].copy()
    lm_s2 = ols(formula, temp_data2).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data2['Pred'] = lm_s2.predict(temp_data2)
    temp_data2['resid'] = lm_s2.resid
    temp_data2.plot('Pred', 'resid', kind='scatter')
    lm_s2.summary()


# 2.2、强影响点分析 更多指标： statemodels包提供了更多强影响点判断指标 （太耗时，最好不要用了）
def strong_influence_point(Xtrain, Ytrain):
    from statsmodels.formula.api import ols
    from statsmodels.stats.outliers_influence import OLSInfluence

    temp_Y = Ytrain
    temp_data = pd.concat([Xtrain, temp_Y], axis=1)
    cols = list(temp_data.columns)
    cols.remove("Y")
    cols_noti = cols
    formula = "Y" + '~' + '+'.join(cols_noti)

    lm_s = ols(formula, data=temp_data).fit()
    OLSInfluence(lm_s).summary_frame().head()


# 解释变量X 之间不能强线性相关 （膨胀系数）
# 3、方差膨胀因子
'''
3.1、两个特征检验：
['Income_ln', 'dist_avg_income_ln']，得到：
Income_ln           36.439462480880216
dist_avg_income_ln  36.439462480880216
> 10 表示 该变量多重共线性严重 （两个特征检验，一定是都大于或都小于）

3.2、三个特征检验：
['Income_ln', 'dist_home_val_ln','dist_avg_income_ln']，得到：
Income_ln           36.653639058963186
dist_home_val_ln    1.053596313570258
dist_avg_income_ln  36.894876856102

Income_ln ~ dist_home_val_ln + dist_home_val_ln > 10  说明Income_ln 与 这两特征 存在多重共线性
dist_avg_income_ln ~ Income_ln + dist_home_val_ln > 10  说明dist_avg_income_ln 与 这两特征 存在多重共线性
dist_home_val_ln<10 ~ Income_ln + dist_avg_income_ln < 10  说明dist_home_val_ln 与 这两特征 不存在多重共线性
所以 Income_ln 和 dist_avg_income_ln 存在多重共线性严重。

如果 特征更多，则需要 选多个特征筛选之后，再进一步小范围筛选。
'''


def vif(df, col_i):
    from statsmodels.formula.api import ols

    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    print(formula)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2), formula


def variance_expansion_coefficient(df, cols):
    temp_df = df[cols].copy()
    temp_dict = {}
    temp_dict_ln = {}

    for i in temp_df.columns:
        temp_v = vif(df=temp_df, col_i=i)
        temp_dict[temp_v[1]] = temp_v[0]
        print(i, '\t', temp_v[0])
        print()

    print("-" * 30)

    logarithm(temp_df, temp_df.columns)
    col_list_ln = [i + "_ln" for i in cols]
    temp_df = temp_df[col_list_ln]
    for i in temp_df.columns:
        temp_v = vif(df=temp_df, col_i=i)
        temp_dict_ln[temp_v[1]] = temp_v[0]
        print(i, '\t', temp_v[0])
        print()

    return temp_dict, temp_dict_ln


# 训练集 与 测试集 拟合：
def fitting_comparison(y_true, y_predict):
    plt.plot(range(len(y_true)), sorted(y_true), c="black", label="Data")
    plt.plot(range(len(y_predict)), sorted(y_predict), c="red", label="Predict")
    plt.legend()
    plt.show()


# 线性回归模型 泛化能力：
def linear_model_comparison(X, y, cv_customize=5, start=1, end=1001, step=100, linear_show=True):
    from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso
    from sklearn.model_selection import cross_val_score

    alpharange = np.arange(start, end, step)
    linear_r2_scores, ridge_r2_scores = [], []
    linear_r2var_scores, ridge_r2var_scores = [], []
    linear_ge, ridge_ge = [], []
    for alpha in alpharange:
        linear = LR()
        ridge = Ridge(alpha=alpha)

        linear_score = cross_val_score(linear, X, y, cv=cv_customize, scoring="r2")
        ridge_score = cross_val_score(ridge, X, y, cv=cv_customize, scoring="r2")

        # 因 R^2=(-∞,1]， R^2拟合优度： 模型捕获到的信息量 占 真实标签中所带的信息量的比例
        # 1 - R^2均值 = 偏差， 所以用 R^2均值 代表偏差（R^2均值越小，偏差越大； R^2均值越大，偏差越小）
        # 偏差：交叉验证 的 R^2均值：不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 均值： 不同模型R^2拟合优度的准确性
        ridge_r2_score = ridge_score.mean()  # R^2均值 代表 偏差
        linear_r2_score = linear_score.mean()
        title_mean = "R2_Mean"
        ridge_r2_scores.append(ridge_r2_score)
        linear_r2_scores.append(linear_r2_score)

        # 方差：交叉验证 的 R^2方差：不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 方差： 不同模型R^2拟合优度的离散程度
        ridge_r2var_score = ridge_score.var()  # R^2 方差
        linear_r2var_score = linear_score.var()
        title_var = "R2_Var"
        ridge_r2var_scores.append(ridge_r2var_score)
        linear_r2var_scores.append(linear_r2var_score)

        # 计算泛化误差的可控部分
        ridge_ge.append((1 - ridge_r2_score) ** 2 + ridge_r2var_score)
        linear_ge.append((1 - linear_r2_score) ** 2 + linear_r2var_score)

    maxR2_Alpha, maxR2 = alpharange[ridge_r2_scores.index(max(ridge_r2_scores))], max(ridge_r2_scores)
    start_Alpha, start_R2 = alpharange[0], ridge_r2_scores[0]
    diff_R2 = maxR2 - start_R2
    print("R^2起始阈值%f:R^2起始值%f，R^2最大值阈值%f:R^2最大值%f，R^2差值%f" % (start_Alpha, start_R2, maxR2_Alpha, maxR2, diff_R2))

    # 当R^2最大值时，求 R^2方差Var的最大值，用R^2的变化差值 与 R^2方差的变化差值 再进行比较
    start_R2VaR = ridge_r2var_scores[0]
    R2VarR_Index = alpharange.tolist().index(maxR2_Alpha)
    R2varR = ridge_r2var_scores[R2VarR_Index]
    diff_R2varR = R2varR - start_R2VaR
    print("R^2方差起始阈值%f:R^2方差起始值%f，R^2方差对应阈值%f:R^2方差对应最大值%f，R^2方差差值%f" % (
    alpharange[0], start_R2VaR, maxR2_Alpha, R2varR, diff_R2varR))
    print("R^2方差差值/R^2差值 = %f" % (diff_R2varR / diff_R2))

    # 1、打印R2最高所对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(alpharange[ridge_r2_scores.index(max(ridge_r2_scores))], max(ridge_r2_scores),
          ridge_r2var_scores[ridge_r2_scores.index(max(ridge_r2_scores))])
    # 1、打印R2方差最低时对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(alpharange[ridge_r2var_scores.index(min(ridge_r2var_scores))],
          ridge_r2_scores[ridge_r2var_scores.index(min(ridge_r2var_scores))], min(ridge_r2var_scores))
    # 1、打印泛化误差可控部分的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(alpharange[ridge_ge.index(min(ridge_ge))], ridge_r2_scores[ridge_ge.index(min(ridge_ge))],
          ridge_r2var_scores[ridge_ge.index(min(ridge_ge))], min(ridge_ge))

    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2_scores, color="red", label="Ridge")
    if linear_show:
        plt.plot(alpharange, linear_r2_scores, color="orange", label="LR")
    plt.title(title_mean)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2var_scores, color="red", label="Ridge")
    if linear_show:
        plt.plot(alpharange, linear_r2var_scores, color="orange", label="LR")
    plt.title(title_var)
    plt.legend()
    plt.show()

    # R2_Var值非常小，从0.0038→0.0045平稳缓慢增加，所以看不出变宽的痕迹。
    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2_scores, c="k", label="R2_Mean")
    plt.plot(alpharange, ridge_r2_scores + np.array(ridge_r2var_scores), c="red", linestyle="--", label="R2_Var")
    plt.plot(alpharange, ridge_r2_scores - np.array(ridge_r2var_scores), c="red", linestyle="--")
    plt.legend()
    plt.title("R2_Mean vs R2_Var")
    plt.show()

    # 绘制 化误差的可控部分
    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_ge, c="gray", linestyle='-.')
    plt.title("Generalization error")
    plt.show()


# In[]:
# ================================线性回归特征分析==============================


# In[]:
# ====================================学习曲线==================================
# In[]:
# -----------------------------1、基于样本量-------------------------------
# 基于MSE绘制学习曲线（样本量）
def plot_learning_curve_mse_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(MSE(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(MSE(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)],
             np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],
             np.sqrt(test_score), label="test")
    plt.legend()
    plt.show()


# 基于R^2值绘制学习曲线（样本量）
def plot_learning_curve_r2_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(r2_score(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(r2_score(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)],
             train_score, label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],
             test_score, label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, -0.1, 1.1])
    plt.show()


# 基于learning_curve函数：
def plot_learning_curve(estimator, title, X, y, scoring=None,
                        ax=None,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    # learning_curve如果不显示设置scoring，则会按照模型默认的指标返回
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , scoring=scoring
                                                            #                                                            ,shuffle=True
                                                            , cv=cv
                                                            , random_state=420
                                                            , n_jobs=n_jobs)

    # 普通交叉验证：
    # 矩阵为5行：与 样本量阈值 相同
    # 5列：折数
    train_scores_mean = np.mean(train_scores, axis=1)  # 按列（折数）取均值
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)  # 按列（折数）取均值
    test_scores_std = np.std(test_scores, axis=1)
    print("样本量阈值%s" % train_sizes)
    print("交叉验证训练集阈值%d,最大分数%f" % (
    train_sizes[test_scores_mean.tolist().index(np.max(test_scores_mean))], np.max(test_scores_mean)))

    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")
    ax.legend(loc="best")
    return ax


# In[]:
# -----------------------------1、基于样本量-------------------------------

# In[]:
# -----------------------------2、基于超参数-------------------------------
# SKLearn库的XGBoost：
def getModel(i, model_name, hparam_name, prev_hparam_value, random_state):
    from xgboost import XGBRegressor as XGBR

    if model_name == "XGBR":
        if hparam_name == "n_estimators":
            reg = XGBR(n_estimators=i, random_state=random_state)
        elif hparam_name == "subsample" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value[0], subsample=i, random_state=random_state)
        elif hparam_name == "learning_rate" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value[0], subsample=prev_hparam_value[1], learning_rate=i,
                       random_state=random_state)
        elif hparam_name == "gamma" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value[0], subsample=prev_hparam_value[1],
                       learning_rate=prev_hparam_value[2], gamma=i, random_state=random_state)
        else:
            raise RuntimeError('Hparam Error')
    return reg


def learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, model_name="XGBR", hparam_name="n_estimators",
                                prev_hparam_value=None, random_state=420):
    rs = []
    var = []
    ge = []
    for i in axisx:
        reg = getModel(i, model_name, hparam_name, prev_hparam_value, random_state)
        cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
        # 记录1-偏差
        rs.append(cvresult.mean())
        # 记录方差
        var.append(cvresult.var())
        # 计算泛化误差的可控部分
        ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
    # 1、打印R2最高所对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
    # 1、打印R2方差最低时对应的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
    # 1、打印泛化误差可控部分的参数取值； 2、并打印这个参数下的R2； 3、并打印这个参数下的R2方差
    print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))

    rs = np.array(rs)
    var = np.array(var)
    plt.figure(figsize=(20, 5))
    plt.plot(axisx, rs, c="black", label=model_name)
    # 添加 方差
    plt.plot(axisx, rs + var, c="red", linestyle='-.')
    plt.plot(axisx, rs - var, c="red", linestyle='-.')
    plt.legend()
    plt.show()

    # 绘制 化误差的可控部分
    plt.figure(figsize=(20, 5))
    plt.plot(axisx, ge, c="gray", linestyle='-.')
    plt.show()


# ---------------------------------------------------------------------------

# 自定义交叉验证（XGBoost原生库）
def learning_curve_xgboost_customize(axisx, X, y, ss, param_fixed, param_cycle_name, num_round):
    import xgboost as xgb

    rs_all_train = []
    var_all_train = []
    ge_all_train = []
    mse_all_train = []
    rs_all_test = []
    var_all_test = []
    ge_all_test = []
    mse_all_test = []

    print(ss.get_n_splits(X))
    print(ss)

    for i in axisx:
        rs_train = []
        mse_train = []
        rs_test = []
        mse_test = []
        param_fixed[param_cycle_name] = i
        print(param_fixed)

        for train_index, test_index in ss.split(X, y):
            #        print("Train Index:", train_index, ",Test Index:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            dtrain = xgb.DMatrix(X_train, y_train)
            bst = xgb.train(param_fixed, dtrain, num_round)
            y_predict_train = bst.predict(dtrain)
            rs_train.append(r2_score(y_train, y_predict_train))
            mse_train.append(MSE(y_train, y_predict_train))

            dtest = xgb.DMatrix(X_test, y_test)
            y_predict_test = bst.predict(dtest)
            rs_test.append(r2_score(y_test, y_predict_test))
            mse_test.append(MSE(y_test, y_predict_test))

        rs_mean_train = np.mean(rs_train)
        rs_var_train = np.var(rs_train)
        rs_all_train.append(rs_mean_train)
        var_all_train.append(rs_var_train)
        ge_all_train.append((1 - rs_mean_train) ** 2 + rs_var_train)
        mse_all_train.append(np.mean(mse_train))

        rs_mean_test = np.mean(rs_test)
        rs_var_test = np.var(rs_test)
        rs_all_test.append(rs_mean_test)
        var_all_test.append(rs_var_test)
        ge_all_test.append((1 - rs_mean_test) ** 2 + rs_var_test)
        mse_all_test.append(np.mean(mse_test))

    print(axisx[rs_all_test.index(max(rs_all_test))], max(rs_all_test),
          var_all_test[rs_all_test.index(max(rs_all_test))])
    print(axisx[var_all_test.index(min(var_all_test))], rs_all_test[var_all_test.index(min(var_all_test))],
          min(var_all_test))
    print(axisx[ge_all_test.index(min(ge_all_test))], rs_all_test[ge_all_test.index(min(ge_all_test))],
          var_all_test[ge_all_test.index(min(ge_all_test))], min(ge_all_test))

    # R2均值、R2方差
    plt.figure(figsize=(20, 5))
    rs_all_train = np.array(rs_all_train)
    var_all_train = np.array(var_all_train)
    plt.plot(axisx, rs_all_train, c="blue", label="XGB_train")
    plt.plot(axisx, rs_all_train + var_all_train, c="green", linestyle='-.')
    plt.plot(axisx, rs_all_train - var_all_train, c="green", linestyle='-.')

    rs_all_test = np.array(rs_all_test)
    var_all_test = np.array(var_all_test)
    plt.plot(axisx, rs_all_test, c="black", label="XGB_test")
    plt.plot(axisx, rs_all_test + var_all_test, c="red", linestyle='-.')
    plt.plot(axisx, rs_all_test - var_all_test, c="red", linestyle='-.')
    plt.title("R2")
    plt.legend()
    plt.show()

    # 绘制 化误差的可控部分
    plt.figure(figsize=(20, 5))
    plt.plot(axisx, ge_all_train, c="blue", label="XGB_train", linestyle='-.')
    plt.plot(axisx, ge_all_test, c="red", label="XGB_test", linestyle='-.')
    plt.title("Ge")
    plt.legend()
    plt.show()

    # MSE
    plt.figure(figsize=(20, 5))
    mse_all_train = np.array(mse_all_train)
    mse_all_test = np.array(mse_all_test)
    plt.plot(axisx, mse_all_train, c="blue", label="XGB_train")
    plt.plot(axisx, mse_all_test, c="red", label="XGB_test")
    plt.title("MSE")
    plt.legend()
    plt.show()


# xgboost原生交叉验证类： xgboost.cv
def learning_curve_xgboost(X, y, param1, param2, num_round, metric, n_fold):
    import xgboost as xgb

    dfull = xgb.DMatrix(X, y)  # 为了便捷，使用全数据

    # X轴一定是num_round：树的数量。 Y轴：回归默认均方误差；分类默认error
    time0 = time()
    cvresult1 = xgb.cv(param1, dfull, num_boost_round=num_round, metrics=(metric), nfold=n_fold)
    # print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

    time0 = time()
    cvresult2 = xgb.cv(param2, dfull, num_boost_round=num_round, metrics=(metric), nfold=n_fold)
    # print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

    plt.figure(figsize=(20, 5))
    plt.grid()
    end_temp = num_round + 1
    plt.plot(range(1, end_temp), cvresult1.iloc[:, 2], c="red", label="train,gamma=0")
    plt.plot(range(1, end_temp), cvresult1.iloc[:, 0], c="orange", label="test,gamma=0")
    plt.plot(range(1, end_temp), cvresult2.iloc[:, 2], c="green", label="train,gamma=20")
    plt.plot(range(1, end_temp), cvresult2.iloc[:, 0], c="blue", label="test,gamma=20")
    plt.legend()
    plt.show()


# In[]:
# -----------------------------2、基于超参数-------------------------------
# In[]:
# ====================================学习曲线==================================


# In[]:
# ====================================交叉验证==================================
# 线性回归：
def rmsle_cv(model, train_X, train_y, cv=None, cv_type=1):
    if cv is None:
        if cv_type == 1:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 交叉验证模式
        elif cv_type == 2:
            cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)
        else:
            raise Exception('CV Type is Error')

    rmse = np.sqrt(-CVS(model, train_X, train_y, scoring="neg_mean_squared_error", cv=cv))
    return (rmse)


# In[]:
# ====================================交叉验证==================================


# In[]:
# =============================Stacking models：堆叠模型=========================
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

'''
堆叠模型：Stacking models
最简单的堆叠方法：平均基本模型:Simplest Stacking approach : Averaging base models
我们从平均模型的简单方法开始。 我们建立了一个新类，以通过模型扩展scikit-learn，并进行封装和代码重用（继承inheritance）。
https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)
'''


# Averaged base models class 平均基本模型
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

    # In[]:
# =============================Stacking models：堆叠模型=========================