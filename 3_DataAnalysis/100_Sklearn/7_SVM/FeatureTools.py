# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:48:40 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, cross_val_score as CVS
from sklearn.linear_model import LinearRegression as LR, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import datetime
import os
from time import time
from statsmodels.formula.api import ols
import Tools_customize as tc


# In[]:
# ================================基础操作 开始==============================
def get_sk_metrics():
    import sklearn.metrics as metrics_name
    return sorted(metrics_name.SCORERS.keys())


# In[]:
# 路径设置
def set_file_path(path):
    import os
    os.chdir(path)


def print_path():
    import os
    print(os.path.abspath(os.curdir))  # 当前路径


# 读入数据源
# https://www.cnblogs.com/datablog/p/6127000.html
'''
1、字符串类型 导入：
字段Seriers类型为object，元素类型为str
1.1、自动转换为np.nan 的输入字符串：
空单元格、NA、nan、NaN、null、NULL
则整个 字段Seriers类型 为object； 字符串元素类型为<class 'str'>；
其中的 空单元格、NA、nan、NaN、null、NULL 元素类型全部变为<class 'float'>也就是np.nan
使用 元素 is np.nan 来判断； 注意使用 from math import isnan 的isnan(元素)函数只能判断np.nan的元素（np.nan类型为float），
在str类型元素上使用报异常，所以 “字符串类型 导入” 不适合使用 isnan(元素)函数进行判断。

1.2、不会自动转换为np.nan 的输入字符串：
NAN、na
其相关元素类型为<class 'str'>；
但不会影响 整个字段Seriers类型，以及其余元素类型。
===================================================================
2、日期类型 导入：
2.1、Excel的 时间字段 单元格格式 为 日期格式 或 自定义日期格式。
2.1.1、使用parse_dates=['auth_time']方式指定日期字段，且导入日期数据必须为 字符串时间类型。不能使用dtype={"auth_time":datetime.datetime或pd.Timestamp}方式（无效）。
2.1.1.1、自动转换为pd.lib.NaT 的输入字符串：
空单元格、NA、nan、NaN、null、NULL、NAN、NaT、nat、NAT
则整个 字段Seriers类型 为datetime64[ns]； 日期元素类型为<class 'pandas._libs.tslib.Timestamp'>；
其中的 空单元格、NA、nan、NaN、null、NULL、NAN、NaT、nat、NAT 元素类型全部变为<class 'pandas._libs.tslib.NaTType'>也就是pd.lib.NaT。
只能使用 元素 is pd.lib.NaT 来判断； 不能使用 元素 is np.nan 来判断（pd.lib.NaT 与 np.nan 不是同一个类型）
注意：能使用 DataFrame.isnull().sum() 检测。

2.1.1.2、如果 输入中包含 非日期格式字符串：（PD转换日期失败）
na
其相关元素类型为<class 'str'>；
则整个 字段Seriers类型 变为object，所有元素类型为str，其中包括：
“2.1.1.1”中的 NA、nan、NaN、null、NULL、NAN、NaT、nat、NAT 元素类型全部变为str
其中输入 NA、null、NULL 自动转换为 nan字符串（全部元素类型为str，没有np.nan）
-------------------------------------------------------------------
2.1.2、直接导入，不指定日期字段
则按照 “1、字符串类型 导入” 中情况进行导入。 例如，Excel中 时间字段 输入
单元格格式 为 日期格式： 9/27/2016  转换为→  2017-06-10<class 'str'>
单元格格式 为 自定义日期格式： 4/15/2017  9:21:18 AM  转换为→  2017-04-15 09:21:18<class 'str'>
都转换为了 时间格式字符串（str类型）。其余的np.nan问题都遵循 “1、字符串类型 导入” 中的情况。
*******************************************************************
2.2、Excel的 时间字段 单元格格式 为 常规。
2.2.1、使用parse_dates=['auth_time']方式指定日期字段。。。未知
-------------------------------------------------------------------
2.2.2、直接导入，不指定日期字段： 按照 “3、数字类型 导入” 的规则进行导入
如果Execl中值为20160712，则导入后转换为 <class 'numpy.int64'>类型 20160712
如果Execl中值为20160217.0，则导入后转换为 <class 'numpy.float64'>类型 20160217.0
===================================================================
3、数字类型 导入：
3.1、显示指定 dtype = {"tail_num":np.float64} 以 np.float64 数据格式导入，才能接受 “空表示字符串”
3.1.1、自动转换为 numpy.float64类型nan 的输入字符串：
空单元格、NA、nan、NaN、null、NULL
则整个 字段Seriers类型 为float64； 数字元素类型为<class 'numpy.float64'>；
其中的 空单元格、NA、nan、NaN、null、NULL 元素类型全部变为<class 'numpy.float64'>也就是numpy.float64类型nan； 但不是np.nan（<class 'float'>）
不能使用 元素 is np.nan 来判断； 只能使用 from math import isnan 的isnan(元素)函数来判断是否为空。
注意：能使用 DataFrame.isnull().sum() 检测。

3.1.2、如果 输入中包含 非数字格式字符串：（PD转换数字失败）
NAN、na 或 字符串“-” 
其相关元素类型为<class 'str'>；
则整个 字段Seriers类型 变为object；
3.1.2.1、输入字符串 NA、nan、NaN、null、NULL 转换为 np.nan。
3.1.2.2、剩下的所有元素类型为str，包括：NAN、na、9753（字符串类型数字）
-------------------------------------------------------------------
3.2、显示指定 dtype = {"tail_num":np.int64} 以 np.intXX 数据格式导入，不能接受 “空表示字符串”
空单元格、NA、nan、null、NULL等 直接报异常。
-------------------------------------------------------------------
3.3、不显示指定 dtype = {"tail_num":np.float64} 或 dtype = {"tail_num":np.int64} 导入格式， Pandas自动选择导入格式：
3.3.1、如果 特征值为：0,1,2,3... 
3.3.1.1、如果 特征中没有空单元格，导入后特征类型为 np.int64
3.3.1.2、如果 特征中有空单元格，导入后特征类型为 np.float64 （因为Pandas自动选择以 np.float64 数据格式导入，才能接受 “空表示字符串”）

3.3.2、如果 特征值为：12.5,12,15...
特征中 有 或 没有 空单元格，导入后特征类型都为 np.float64 （因为Pandas自动选择以 np.float64 数据格式导入，才能接受 “空表示字符串”）

3.3.3、np.float64 和 np.int64 的情况遵循： 3.1 和 3.2 中的描述。
===================================================================
'''


def readFile_inputData(train_name=None, test_name=None, index_col=None, dtype=None, parse_dates=None, encoding="UTF-8",
                       sep=','):
    if parse_dates is not None and type(parse_dates) != list:
        raise Exception('parse_dates Type is Error, must list')
    if train_name is not None:
        train = pd.read_csv(filepath_or_buffer=train_name, index_col=index_col, dtype=dtype, parse_dates=parse_dates,
                            encoding=encoding, sep=sep)
    if test_name is not None:
        test = pd.read_csv(filepath_or_buffer=test_name, index_col=index_col, dtype=dtype, parse_dates=parse_dates,
                           encoding=encoding, sep=sep)
        return train, test
    else:
        return train


# 关于low_memory参数： pandas read_csv mixed types问题： https://www.jianshu.com/p/a70554726f26


# 没有表头的导入方式：
'''
header=None： 文件中没有列名这一行，使用names指定的列名。
header=0： 使用文件中的第0行作为列名。
header=0与names共用：丢弃文件中的第0行，使用names指定的列名。 header=1与names共用：丢弃文件中的第0、1两行，使用names指定的列名。

tags = ft.readFile_inputData_no_header('tags.csv', ["uid", "mid", "tag", "timestamp"])
'''


def readFile_inputData_no_header(path, names=None, header=None, index_col=None, dtype=None, na_values=None,
                                 usecols=None, encoding="UTF-8", sep=','):
    if names is not None and type(names) != list:
        raise Exception('names Type is Error, must list')
    return pd.read_csv(filepath_or_buffer=path, names=names, header=header, index_col=index_col, dtype=dtype,
                       na_values=na_values, usecols=usecols, encoding=encoding, sep=sep)


# 保存数据
def writeFile_outData(data, path, encoding="UTF-8", index=False):
    data.to_csv(path, encoding=encoding, index=index)


# 原生读取.txt文件（只是个示例）
# item_info = get_item_info('movies.txt') 需先调用set_file_path设置路径
def get_item_info(input_file, split_char="::", title_num=None, encoding="UTF-8"):
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    fp = open(input_file, encoding=encoding)
    line_num = 0
    for line in fp:
        if (title_num is not None) and (line_num <= title_num):
            line_num += 1
            continue
        item = line.strip().split(split_char)
        if len(item) < 3:
            continue
        elif len(item) == 3:
            itemId, title, genre = item[0], item[1], item[2]
        else:
            itemId = item[0]
            genre = item[-1]
            title = ",".join(item[1:-1])
        item_info[itemId] = [title, genre]
    fp.close
    return item_info


# 读取数据 转化为 二分图数据结构
def get_graph_from_data(input_file, split_char="::", score_thr=3.0, title_num=None, encoding="UTF-8"):
    '''
    Args:
        input_file: user item rating file
    Return:
        a dict: {UserA:{itemb:1, itemc:1}, itemb:{UserA:1}}
    '''
    if not os.path.exists(input_file):
        return {}
    graph = {}
    fp = open(input_file, encoding=encoding)
    line_num = 0
    for line in fp:
        if (title_num is not None) and (line_num <= title_num):
            line_num += 1
            continue
        item = line.strip().split(split_char)
        if len(item) < 3:
            continue
        userid, itemid, rating = item[0], "item_" + item[1], item[2]
        if float(rating) < score_thr:
            continue
        if userid not in graph:
            graph[userid] = {}  # 字典套字典
        graph[userid][itemid] = 1
        if itemid not in graph:
            graph[itemid] = {}
        graph[itemid][userid] = 1
    fp.close
    return graph


# 合并数据源（列向）
def consolidated_data_col(train_X, train_y, axis=1):
    return pd.concat([train_X, train_y], axis=axis)


# 分离数据源（列向）
def separate_data_col(df, y_name):
    train_X = df.drop(y_name, axis=1)
    train_y = pd.DataFrame(df[y_name])
    return train_X, train_y


# 合并数据源（行向）： 训练集 与 测试集 合并
# 注意： 在合并train和test数据之前，先将训练集train的 异常、离群值 相应行数据删除（测试集test不能删除行数据）
def consolidated_data_row(train, test, y_name):
    ntrain = train.shape[0]
    ntest = test.shape[0]
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    all_data_X, train_y = separate_data_col(all_data, y_name)
    train_y = train_y[:ntrain]
    print("all_data size is : {}".format(all_data.shape))
    return all_data, all_data_X, train_y, ntrain, ntest


# 分离数据源（行向）： 训练集 与 测试集 分离
'''
坑：
all_data[:ntrain] 是按照 :ntrain 的数量 取值： 0 → ntrain-1
all_data.loc[:ntrain,y_name] 是按照 :ntrain 的 行索引名 取值： 0 → ntrain
'''


def separation_data_row(all_data, ntrain, y_name=None):
    if y_name is None:
        train_X = all_data[:ntrain]
        test_X = all_data[ntrain:]
        return train_X, test_X
    else:
        train_y = pd.DataFrame(all_data.loc[:ntrain - 1, y_name])
        all_data = all_data.drop(y_name, axis=1)
        train_X = all_data[:ntrain]
        test_X = all_data[ntrain:]
        return train_X, test_X, train_y


# 更改列名
def seriers_change_colname(seriers, col_name):
    seriers.name = col_name


def seriers_change_index(seriers, index):
    seriers.index = index


def df_change_colname(df, columns, inplace=True):
    if type(columns) != dict:
        raise Exception('columns Type is Error, must dict')
    df.rename(columns=columns, inplace=inplace)


# train_data.drop(train_data[train_data['price'] <= 0].index, axis=0, inplace=True)
def simple_drop_data(data, condition=None, cols_name=None, axis=0, inplace=True):
    if axis == 0 and condition is not None:
        data.drop(condition, axis=0, inplace=inplace)  # condition应是行索引
        recovery_index(data)
    elif axis == 1 and condition is None and cols_name is not None:
        data.drop(cols_name, axis=1, inplace=inplace)


# 恢复索引（删除数据后：如果X集恢复了索引，那么Y集也必须恢复索引）
def recovery_index(data_list):
    if type(data_list) == list:
        for i in data_list:
            i.index = range(i.shape[0])
    #            i = i.reset_index(drop=True)
    else:
        data_list.index = range(data_list.shape[0])


# 排序
def data_sort(data, sort_cols, ascendings, inplace=False):
    if type(sort_cols) != list:
        raise Exception('sort_cols Type is Error, must list')
    elif type(ascendings) != list:
        raise Exception('ascendings Type is Error, must list')

    return data.sort_values(by=sort_cols, ascending=ascendings, inplace=inplace)


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
def astype_customize(x, t):
    try:
        return t(x)
    except:
        return np.nan


# 设置特征类型： （所有的 分类类别 都使用str，而不使用 categorical类型（有的库报异常））
def set_classif_col(df, feature_name, val_type=1):
    if val_type == 1:
        temp_type = int
    elif val_type == 2:
        temp_type = float
    elif val_type == 3:
        temp_type = str
    else:
        raise Exception('Val Type is Error')

    # np.nan可以用于astype()函数； 字符串"nan" 在Seriers的数据类型为str时 和 np.nan 在赋值时等价
    try:
        df[feature_name] = df[feature_name].astype(temp_type)
    except:
        df[feature_name] = df[feature_name].map(lambda x: astype_customize(x, temp_type))


# float16/float32/float64： 其中有空值 <class 'numpy.float64'>也就是numpy.float64类型nan； 但不是np.nan（<class 'float'>）
# 如果直接str(x)转换，则nan会转换为： 字符串'nan'，所以只能如下操作。 int类型暂时还不知道是什么情况。
def num_to_char(df, feature_name):
    from math import isnan
    df[feature_name] = df[feature_name].map(lambda x: np.nan if isnan(x) else str(x))


def feature_category(data):
    # 数字特征
    numeric_features = data.select_dtypes(include=[np.number])
    # 类型特征
    categorical_features = data.select_dtypes(include=[np.object])
    return numeric_features.columns, categorical_features.columns


# 特征类型转换 以 减少内存消耗： （相传是祖传代码 (⊙o⊙)）
def reduce_mem_usage(df, is_set_category=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif is_set_category:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[]:
# 缺失值：行向（特征缺失值）、列向（数据缺失值）； 1:删特征、 2:删数据
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

被除数为0， 除数为0： float64数据类型的 商 为numpy.float64类型nan。
被除数非0， 除数为0： float64数据类型的 商 为inf正无穷。 所以除数字段的缺失值在相除计算之后，再填充。
'''


def missing_values_table(df, customize_axis=0, percent=None, del_type=1):
    # Total missing values
    # customize_axis=0： 按行向统计： 一列有多少缺失值
    # customize_axis=1： 按列向统计： 一行有多少缺失值
    mis_val = df.isnull().sum(axis=customize_axis)

    # Percentage of missing values
    mis_val_percent = 100 * mis_val / df.shape[customize_axis]

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing_Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    # index索引 就是 特征名称
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    if customize_axis == 0 and percent is not None:
        if del_type == 1:  # 删特征列
            temp_drop_col = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] > percent].index.tolist()
            df_nm = df.copy()
            df_nm.drop(temp_drop_col, axis=1, inplace=True)
            return mis_val_table_ren_columns, df_nm
        elif del_type == 2:  # 删数据行
            temp_drop_col = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] > percent].index.tolist()
            df_nm = df.copy()
            for i in temp_drop_col:
                df_nm.drop(df_nm.loc[df_nm[i].isnull()].index, axis=0, inplace=True)
            # 恢复索引（删除行数据 需要恢复索引）
            recovery_index([df_nm])
            return mis_val_table_ren_columns, df_nm
        else:
            return mis_val_table_ren_columns
    elif customize_axis == 1 and percent is not None:
        temp_drop_row = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] > percent].index.tolist()
        df_nm = df.copy()
        df_nm.drop(temp_drop_row, axis=0, inplace=True)
        # 恢复索引（删除行数据 需要恢复索引）
        recovery_index([df_nm])
        return mis_val_table_ren_columns, df_nm

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


'''
缺失值特征X 对 因变量Y 的影响： （二分类）
1、先分析 缺失值特征 对 因变量Y 的影响： missing_values_2categories_compare函数，通过 缺失值百分比 找到 对因变量Y 影响较大的 缺失值特征（Y==1时缺失值百分比 明显高于 Y==0时缺失值百分比）。

2、再对 “1” 中找到的 缺失值特征 进行 再细化分析： feature_missing_value_analysis函数。
看 04094_my.py 中 “5.1、为订单表建立临时表（目标1：求 缺失值 对 违约率 的影响； 目标2：创建 新特征）”
2.1、细化到 缺失值特征A：
2.1.1、当 以每个用户为分组条件 多数样本出现 缺失值特征A 的累加缺失值很大，但因变量Y不是违约状态时，且违约比很低； 缺失值特征A 对 预测因变量Y 没有效果。
2.1.2、当 以每个用户为分组条件 多数样本出现 缺失值特征A 的累加缺失值很大，因变量Y是违约状态时，缺失值特征A 对 预测因变量Y 有效果。
2.1.2.1、当以 缺失值特征A 为主要分析对象时，其缺失值百分比为100%，如果 其他缺失值特征 的缺失值百分比 也几乎接近于100%，
那么新增 缺失值特征A标识 对 预测因变量Y 效果不明显（共线性）。从其中选择部分 缺失值特征 做 新增 缺失值特征标识。
2.1.2.2、当以 缺失值特征A 为主要分析对象时，其缺失值百分比为100%，如果 其他缺失值特征 的缺失值百分比 远低于100%，那么新增 缺失值特征A标识 对 预测因变量Y 有效果。

2.2、细化到 缺失值特征B：
2.2.1、当 以每个用户为分组条件 多数样本出现 缺失值特征B 的累加缺失值很大，因变量Y是违约状态时，缺失值特征B 对 预测因变量Y 有效果。
2.2.1.1、当以 缺失值特征B 为主要分析对象时，其缺失值百分比为100%，如果 缺失值特征A 的缺失值百分比 远低于100%，
而 缺失值特征B 的 违约比 原大于 缺失值特征A 的 违约比， 则 违约比多出来的部分 就是 缺失值特征B 比 缺失值特征A 多出的缺失值 给出的。 
（前提条件：当以 缺失值特征A 为主要分析对象时，其缺失值百分比为100%， 缺失值特征B 的缺失值百分比 几乎接近于100%）
'''


def missing_values_2categories_compare(df, y_name):
    tmp_null_target_1 = missing_values_table(df[df[y_name] == 1])
    tmp_null_target_1 = tmp_null_target_1.reset_index()
    tmp_null_target_1.rename(columns={"index": "feature", "Missing Values": "Missing Values_target_1",
                                      "% of Total Values": "% of Total Values_target_1"}, inplace=True)
    tmp_null_target_0 = missing_values_table(df[df[y_name] == 0])
    tmp_null_target_0 = tmp_null_target_0.reset_index()
    tmp_null_target_0.rename(columns={"index": "feature", "Missing Values": "Missing Values_target_0",
                                      "% of Total Values": "% of Total Values_target_0"}, inplace=True)
    tmp_null_target_merge = pd.merge(tmp_null_target_1, tmp_null_target_0, on=['feature'])
    return tmp_null_target_merge


def feature_missing_value_analysis(df, feature_name, groupby_col, y_name):
    tmp_df_f_null = df[df[feature_name].isnull()]
    f_null_stat = missing_values_table(tmp_df_f_null)
    agg = {'f_null_count': len, 'f_null_target': np.mean}  # （速度近10倍于groupby_apply）
    f_null_stat2 = tc.groupby_agg_oneCol(tmp_df_f_null, [groupby_col], y_name, agg, as_index=False)
    target1_sum = f_null_stat2[f_null_stat2["f_null_target"] == 1]["f_null_count"].sum()
    target0_sum = f_null_stat2[f_null_stat2["f_null_target"] == 0]["f_null_count"].sum()
    f_null_ratio = target1_sum / target0_sum  # 违约比

    return tmp_df_f_null, f_null_stat, f_null_stat2, f_null_ratio


# 特征返回非缺失值部分
def get_notMissing_values(data_temp, feature):
    if type(feature) == list:
        raise Exception('feature Type is Error, must not list')
    # 注意： 切片操作产生一个新对象了，地址当然不同（即使显示的取所有列，也是新地址）。 参看 Python笔记： “赋值与地址”
    # 就算 data_temp = data_temp[data_temp[feature] == data_temp[feature]] data_temp现在也是新对象，地址不同。
    return data_temp[data_temp[feature] == data_temp[feature]]  # 返回全部Data


'''
1、缺失值填充：不用“多重差补”。用均值或中位数填充，用模型开发时的均值填充，而不是运行时数据的均值填充。
2、要保证模型数据的稳定性，最担心的就是变量飘逸：模型运行时X的均值发生变化，随之预测Y的均值也发生改变，模型就不起效了。
'''


# 缺失值填充
def missValue_all_fillna(df, nan_cols=None, fill_value=None, fillna_type=1):
    if fillna_type == 1:
        if nan_cols is None or type(nan_cols) != list:
            raise Exception('nan_cols Type is Error, must list')
        elif fill_value is None:
            raise Exception('fill_value is Null')
        for i in nan_cols:
            df[i] = df[i].fillna(fill_value)
    else:
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


# 空单元格读取进来 是 np.nan
# 缺失值填充0/1值
def missValue_map_fillzo(df, nan_col, nan_type=1, null_val=0, non_null_val=1):
    from math import isnan

    if nan_type == 1:  # str
        if non_null_val is None:
            f = lambda x: null_val if x is np.nan else x
        else:
            f = lambda x: null_val if x is np.nan else non_null_val
    elif nan_type == 2:  # 日期
        if non_null_val is None:
            f = lambda x: null_val if x is pd.lib.NaT else x
        else:
            f = lambda x: null_val if x is pd.lib.NaT else non_null_val
    else:  # 浮点数
        if non_null_val is None:
            f = lambda x: null_val if isnan(x) else x
        else:
            f = lambda x: null_val if isnan(x) else non_null_val
    return df[nan_col].map(f)  # 缺0有1


# 缺失值 统一转换
def missValue_map_conversion(df, nan_col, type_d=1):
    from math import isnan

    if df[nan_col].dtypes == object:  # 字符串类型
        # 这2个是不能自动转换为 np.nan 的，所以手动
        lam = lambda x: np.nan if ((x == 'na') | (x == 'NAN')) else float(x) if type_d == 1 else x
        return df[nan_col].map(lam)
    #    elif df[nan_col].dtypes == np.float64: # 数字类型(默认float64)
    #        lam = lambda x: np.nan if isnan(x) else float(x) # 将 numpy.float64类型nan 转换为 np.nan<float>： 高位转低位，转换无效。
    #        return df[nan_col].map(lam)
    else:
        return df[nan_col]


# 缺失值 datatime
# 两种数据类型： 1、字符串时间格式； 2、时间戳
def missValue_datatime(df, col, str_format="%Y-%m-%d %H:%M:%S"):
    from math import isnan
    # datetime.datetime.strptime 字符串 转 datetime
    # datetime.datetime.utcfromtimestamp 时间戳 转 datetime
    #  + datetime.timedelta(hours = 8) 向后推8小时
    # 将 np.nan<float> 转换为 numpy.float64类型nan： 低位转高位，转换有效。
    # 且 使用 np.nan<float> 转换为 numpy.float64类型nan 时，使用datetime.datetime的转换 会自动转换为 pd.Timestamp。
    # 且 只要有值转换为 pd.lib.NaT，使用datetime.datetime的转换 会自动转换为 pd.Timestamp。
    return df[col] \
        .map(lambda x: pd.lib.NaT if (x is np.nan or isnan(x) or str(x) == '0' or str(x) == 'na' or str(x) == 'NAN')
    else (datetime.datetime.strptime(str(x), str_format) if ':' in str(
        x)  # 如果是 包含: 的字符串时间格式，则转换为datetime格式。 否则是数字的时间戳，也转换为datetime格式。
    # 使用utcfromtimestamp + timedelta的意义在于 避开系统本地时间的干扰，都可以准确转换到 东八区时间。
    else (datetime.datetime.utcfromtimestamp(int(str(x)[0:10])) + datetime.timedelta(hours=8))
    )
             )


def missValue_datatime_simple(df, col, str_format="%Y%m%d"):
    from math import isnan
    return df[col] \
        .map(lambda x: pd.lib.NaT if (x is np.nan or isnan(x) or str(x) == '0' or str(x) == 'na' or str(x) == 'NAN')
    else (datetime.datetime.strptime(str(x)[0:8], str_format))
             )


# 缺失值 datatime
# 当 剩下的都是 可以正常转换的 字符串时间格式 时。 程序会自动把把re.match匹配到的 字符串时间格式 转换为 pd.Timestamp
def missValue_datatime_match(df, col):
    import re
    return df[col].map(lambda x: x if (re.match("^(19|20)\d{2}-\d{1,2}-\d{1,2}", str(x))) else pd.lib.NaT)


# 自定义时间转换pd.Timestamp：
# 代码在： 1_Used_car_transaction_price_prediction/2_Feature_Engineering.py
def custom_time_conversion(data, time_col_before, time_col_after):
    # 参看 Python笔记： “赋值与地址”
    # 注意： 增加新特征 同样是在入参data上进行操作， 就和 调用变量train 有关系， 不会产生新对象
    # 1、向量API 按特征列 忽略时间格式错误 转换为pd.Timestamp： 错误时间格式，如：19910001，将被置为 pd.lib.NaT
    data[time_col_before + "1"] = pd.to_datetime(data[time_col_before], format='%Y%m%d', errors='coerce')

    # 2、自定义转换：
    # 将 月为00 的错误时间格式，截取到年。
    data[time_col_before + "2"] = data[time_col_before].map(lambda x: x[0:4] if x[4:6] == '00' else x)
    # 并转换为pd.Timestamp
    data[time_col_before + "3"] = data[time_col_before + "2"].map(
        lambda x: pd.to_datetime(x, format="%Y") if (len(x) == 4) else pd.to_datetime(x, format="%Y%m%d"))

    # 3、计算时间差
    # 3.1、方式1： DataFrame.apply()。 差值.days
    data['diff_day2'] = data.apply(lambda x: (x[time_col_after] - x[time_col_before + "3"]).days, axis=1)

    # 3.2、方式2： Seriers直接相减。 注意： 差值.dt.days
    data['diff_day3'] = (data[time_col_after] - data[time_col_before + "3"]).dt.days

    # 3.2.1、注意： 如果 pd.Timestamp - pd.lib.NaT = np.nan
    data['diff_day1'] = (data[time_col_after] - data[time_col_before + "1"]).dt.days
    # 截止到这里都是在 入参data上进行操作， 就和 调用变量train 有关系， 不会产生新对象。

    # 注意： 而 切片操作产生一个新对象了，地址当然不同（即使显示的取所有列，也是新地址）。 所以返回的是新对象。
    return data[
        [time_col_before, time_col_before + "1", time_col_before + "2", time_col_before + "3", 'diff_day1', 'diff_day2',
         'diff_day3']]


# In[:]
# 重复值处理（按行统计）
def duplicate_value(data, subset=None, keep='first', inplace=False):
    #    if type(subset) is not list:
    #        raise Exception('subset Type is Error')
    #    elif len(subset) < 2:
    #        raise Exception('subset size must lager than 2')

    # 去除重复项 后 长度
    nodup = data[-data.duplicated(subset=None, keep=keep)]
    print("去除重复项后长度：%d(按行统计)" % len(nodup))
    # 去除重复项 后 长度
    print("去除重复项后长度：%d(按行统计)" % len(data.drop_duplicates(subset=None, keep=keep)))
    # 重复项 长度
    print("重复项长度：%d(按行统计)" % (len(data) - len(nodup)))

    if inplace:
        # 按行统计，以subset指定的特征列为统计目标
        data.drop_duplicates(subset=subset, keep=keep, inplace=inplace)  # 在原数据集上 删除重复项
        # 重设索引
        recovery_index([data])

    print(data.info())


# 重复索引处理（按行统计）
def duplicate_index(data, keep='first'):
    return data[~data.index.duplicated(keep=keep)]


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


# 所有 连续特征 异常值对比检测：
def all_con_mode_median_iqr_outlier(data, var_c_s):
    if type(var_c_s) != list:
        raise Exception('var_c_s Type is Error, must list')

    # 利用 众数 减去 中位数 的差值  除以  四分位距 来 查找是否有可能存在异常值（案例2：精准营销的两阶段预测模型3 的 9分钟）
    # 原理： 异常值一般表现为众数； 正常值一般表现为中位数； 看两者之间的差值 / 四分位距
    # 如果 特征有两个峰值很突出，造成temp_outlier>0.9很大，这种情况属于正常情况，不用清洗。
    iqr = data[var_c_s].quantile(0.75) - data[var_c_s].quantile(0.25)
    temp_outlier = abs((data[var_c_s].mode().iloc[0,] - data[var_c_s].median()) / iqr)
    print("如果 众数 减去 中位数 的差值 除以 四分位距 的值很大，需要进一步用直方图观测，对嫌疑大的变量进行可视化分析")
    temp_outlier = temp_outlier.sort_values(ascending=False)
    print(temp_outlier)
    return temp_outlier


# 离群值检测： 使用 箱型图、散点趋势图 观测离群值
def outlier_detection(X, feature, y=None, y_name=None, fit_type=1, box_scale=1.5):
    if type(feature) == list:
        # 盒须图 要求 特征必须为单特征，不能传['x']进来
        raise Exception('feature Type is Error, must not list')

    if y is not None and y_name is not None:
        ntrain = y[y_name].notnull().sum()
        X = X[0:ntrain]

    # 利用 众数 减去 中位数 的差值  除以  四分位距来 查找是否有可能存在异常值
    # 如果值很大，需要进一步用直方图观测，对嫌疑大的变量进行可视化分析
    f, axes = plt.subplots(1, 2, figsize=(23, 8))
    box_more_index, hist_more_index = con_data_distribution(X, feature, axes, fit_type=fit_type, box_scale=box_scale)
    # 从直方图中可以看出： 如果数据有最大峰值，属于正常数据，不用清洗。

    # 盒须图的上下限
    upper_more_index = box_more_index[0]
    down_more_index = box_more_index[1]
    #    print(X.iloc[upper_more_index].shape)
    #    print(y.iloc[upper_more_index].shape)

    # 直方图的左右限
    hist_left_more_index = hist_more_index[0]
    hist_right_more_index = hist_more_index[1]

    if y is not None and y_name is not None:
        f, axes = plt.subplots(1, 1, figsize=(23, 8))
        sns.regplot(X[feature], y[y_name], ax=axes)

        f, axes = plt.subplots(1, 2, figsize=(23, 8))
        sns.regplot(X.loc[upper_more_index][feature], y.loc[upper_more_index][y_name], ax=axes[0])
        sns.regplot(X.loc[down_more_index][feature], y.loc[down_more_index][y_name], ax=axes[1])

    # 上下限， 左右限
    return box_more_index, hist_more_index


# 删除离群值
def delete_outliers(X_Seriers, X_name, X_value, y_Seriers, y_name, y_value):
    # 多条件查询方式1：
    #    del_index = X_Seriers.loc[(X_Seriers[X_name]>X_value) & (y_Seriers[y_name]<y_value)].index
    # 多条件查询方式2：
    del_index = X_Seriers.loc[
        (X_Seriers[X_name].map(lambda x: x > X_value)) & (y_Seriers[y_name].map(lambda x: x < y_value))].index
    if id(X_Seriers) == id(y_Seriers):
        X_Seriers.drop(del_index, axis=0, inplace=True)
        # 恢复索引
        recovery_index([X_Seriers])
    else:
        X_Seriers.drop(del_index, axis=0, inplace=True)  # 如果X集恢复了索引，那么Y集也必须恢复索引
        y_Seriers.drop(del_index, axis=0, inplace=True)
        # 恢复索引
        recovery_index([X_Seriers, y_Seriers])


# In[]:
# 分类模型 数据类别 样本不均衡（训练集 与 测试集）
def sample_category(ytest, ytrain):
    train_unique_label, train_counts_label = category_quantity_statistics_all(ytrain)
    test_unique_label, test_counts_label = category_quantity_statistics_all(ytest)
    print('-' * 60)
    print('Label Distributions: \n')
    print("训练集类别%s，数量%s，占比%s" % (train_unique_label, train_counts_label, (train_counts_label / len(ytrain))))
    print("测试集类别%s，数量%s，占比%s" % (test_unique_label, test_counts_label, (test_counts_label / len(ytest))))


# 分类模型 数据类别 样本不均衡可视化（单一数据集测试）
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
    temp_list.append(list(set(set_one) & set(set_two)))  # 交集
    temp_list.append(list(set(set_one) - (set(set_two))))  # 差集 （项在set_one中，但不在set_two中）
    temp_list.append(list(set(set_one) ^ set(set_two)))  # 补-对称差集（项在set_one或set_two中，但不会同时出现在二者中）
    temp_list.append(list(set(set_one) | set(set_two)))  # 并集
    return temp_list


# 设置分类变量类型为：category （在“盒须图”中分类特征 临时 设置为category变量类型）
# 注意： 分类特征设置为category类型后，使用其他库时可能会报错，所以category类型转换 只能 临时 在图表显示中使用，不能修改到原数据。
'''
注意： astype()没有inplace关键字
# 1、方式1： 分两步设置
# 1.1、分类特征类型设置为 category类型
# dat2.dist=dat1.dist.astype("category") 
# 1.2、设置分类特征category类型的显示顺序
# dat2.dist.cat.set_categories(["石景山","丰台","朝阳","海淀","东城","西城"],inplace=True) # 有inplace关键字

# 2、方式2： 使用 .astype('category', categories=categories_) 一步设置
'''


def set_col_category(df, feature_name, categories_=None):
    if categories_ is None:
        df[feature_name] = df[feature_name].astype('category', ordered=True)  # 自动排序： 按首字母顺序
    else:
        df[feature_name] = df[feature_name].astype('category', categories=categories_)  # 手动排序


# 分类特征设置为category类型，为np.nan添加一个类别 （在“盒须图”中分类特征 临时 设置为category变量类型）
def add_col_category(data, categorical_features):
    for c in categorical_features:
        data[c] = data[c].astype('category')
        if data[c].isnull().any():
            # 为有np.nan的分类特征的category类型 添加一个category类别：missing
            data[c] = data[c].cat.add_categories(['missing'])
            data[c] = data[c].fillna('missing')


# 分类变量编码：
# 分类特征 普通编码：
def classif_labelEncoder(data, cols):
    from sklearn.preprocessing import LabelEncoder

    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))


# 分类变量 类别占比 小于阈值 统一 归为 一个类别
def category_col_compress(seriers_col, threshold=0.005, other_name="dum_others"):
    # copy the code from stackoverflow
    dummy_col = seriers_col.copy()

    # what is the ratio of a dummy in whole column
    count = pd.value_counts(dummy_col) / len(dummy_col)

    # cond whether the ratios is higher than the threshold
    mask = dummy_col.isin(count[count > threshold].index)

    # replace the ones which ratio is lower than the threshold by a special name
    dummy_col[~mask] = other_name

    return dummy_col


# 分类变量 独热编码 一般情况下不用 独热编码
def category_getdummies(pre_combined, cat_cols=None, cat_threshold=0):
    if cat_cols is None:
        cat_cols = pre_combined.select_dtypes(include=[np.object]).columns.tolist()

    for col in cat_cols:
        pre_combined[col] = category_col_compress(pre_combined[col],
                                                  threshold=cat_threshold)  # threshold set to zero as it get high core for all estimatior  except ridge based

    dummies_val = pd.get_dummies(pre_combined[cat_cols], prefix=cat_cols)  # 自动加下划线
    pre_combined = pre_combined.join(dummies_val)

    return pre_combined


# 分类变量去除空格
# https://blog.csdn.net/lynxzong/article/details/90552470
def category_remove_spaces(data, cat_cols=None, del_str=None):
    if del_str is not None:
        tmp_f = lambda x: x.strip().replace(del_str, '')  # lambda x : re.sub(",","",x.strip())
    else:
        tmp_f = lambda x: x.strip()

    if type(data) is pd.core.frame.DataFrame:
        if cat_cols is None:
            cat_cols = data.select_dtypes(include=[np.object]).columns.tolist()
        else:
            if type(cat_cols) is not list:
                raise Exception('cat_cols Type is Error, must list')
        data[cat_cols] = data[cat_cols].applymap(tmp_f)
    elif type(data) is pd.core.series.Series:
        return data.map(tmp_f)
    else:
        raise Exception('data Type is Error')


# 分类变量手动编码
def category_manual_coding(data, maps):
    if type(maps) is not dict:
        raise Exception('maps Type is Error')

    if type(data) is pd.core.frame.DataFrame:
        '''
        maps = {"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45"},
                "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May"}}
        train = ft.category_manual_coding(train, maps)
        '''
        return data.replace(maps)
    elif type(data) is pd.core.series.Series:
        '''
        maps = {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45"}
        train["MSSubClass"] = ft.category_manual_coding(train["MSSubClass"], maps)
        '''
        return data.map(maps)
    else:
        raise Exception('data Type is Error')


# 分类变量手动编码： 按分类变量的 每个类别的样本数量来设置离散值的前后顺序，再自定义one-hot编码字符串
def category_customize_coding(train_data, test_data, features=None):
    from sklearn.preprocessing import LabelEncoder

    # 以train训练集的类别为基准
    if features is None:
        features = train_data.select_dtypes(include=[np.object]).columns.tolist()
    if type(features) is not list:
        raise Exception('features Type is Error, must list')

    for feature in features:
        _, _, unique_dict = category_quantity_statistics_all(train_data, feature)
        if len(unique_dict) == 1:
            continue
        elif len(unique_dict) == 2:
            lbl = LabelEncoder()
            train_data[feature] = lbl.fit_transform(list(train_data[feature].values))
            test_data[feature] = lbl.fit_transform(list(test_data[feature].values))
        else:
            output_dict = {}
            tmp_index = 0
            for zuhe in tc.dict_sorted(unique_dict):
                output_dict[zuhe[0]] = tmp_index
                tmp_index += 1

            train_data[feature] = train_data[feature].apply(customize_one_hot, args=(output_dict,))
            test_data[feature] = test_data[feature].apply(customize_one_hot, args=(output_dict,))


def customize_one_hot(x, feature_dict):
    output_list = [0] * len(feature_dict)
    if x in feature_dict:  # else: 如果 test数据集中 没有 train数据集中的类别，就全部置为0。
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])


# In[]:
# 分类变量：统计类别数量： （区分特征） 当为DataFrame时才能使用axis关键字，也可为Seriers。优点能去除缺失值
def category_quantity_statistics(df, features, axis=0, dropna=True):
    if type(features) is list:
        return df[features].nunique(axis=axis, dropna=dropna)  # dropna：bool，默认为True，不在计数中包含np.nan。
    else:
        return df[features].nunique(dropna=dropna)


# 分类变量：统计类别数量： （不区分特征：当为多特征时，不按特征区分，综合统计。（没有axis参数）） 主要为Seriers
# 注意： np.unique统计时会包含： np.nan（一个np.nan为一个类别，很麻烦，没有dropna关键字）； 有np.nan不要使用。
# unique_label, counts_label, unique_dict = ft.category_quantity_statistics_all(Series)
def category_quantity_statistics_all(df, feature=None):
    # unique_label：非重复值集合列表； counts_label：每个非重复值的数量
    if type(df) is pd.core.series.Series and feature is None:
        unique_label, counts_label = np.unique(df, return_counts=True)  # df是Seriers
    elif type(df) is pd.core.frame.DataFrame and feature is not None:
        unique_label, counts_label = np.unique(df[feature], return_counts=True)  # df是DataFrame
    else:
        raise Exception('Type is Error')

    unique_dict = {}
    for i in range(len(unique_label)):
        unique_dict[unique_label[i]] = counts_label[i]
    #        print(unique_label[i], counts_label[i])

    return unique_label, counts_label, unique_dict
    # df[features].value_count().to_dict() 直接就是个dict


# 分类变量：类别统计 (value_counts()会自动剔除np.nan)
def category_quantity_statistics_value_counts(data, category_index, continuous_index=None, index_type=1):
    if type(category_index) != list:
        raise Exception('category_index Type is Error, must list')

    if index_type == 1:
        for i in category_index:
            print("{}特征有个{}不同的值".format(i, category_quantity_statistics(data, i)))
            print(data[i].agg(['value_counts']).T)
            print("=======================================================================")

        if continuous_index is not None:
            print()
            print("-" * 60)
            print("Continuous feature:")
            print("=======================================================================")
            if type(continuous_index) != list:
                raise Exception('continuous_index Type is Error, must list')
            for i in continuous_index:
                print(i, ":")
                print(data[i].agg(['min', 'mean', 'median', 'max', 'std']).T)
                print("=======================================================================")
    else:
        # 列索引数字下标（侬脑子瓦特了）
        for i in category_index:
            print("{}特征有个{}不同的值".format(data.columns.values[i],
                                        category_quantity_statistics(data, data.columns.values[i])))
            print(data[data.columns.values[i]].agg(['value_counts']).T)
            print("=======================================================================")

        if continuous_index is not None:
            print()
            print("-" * 60)
            print("Continuous feature:")
            print("=======================================================================")
            if type(continuous_index) != list:
                raise Exception('continuous_index Type is Error, must list')
            for i in continuous_index:
                print(data.columns.values[i], ":")
                print(data[data.columns.values[i]].agg(['min', 'mean', 'median', 'max', 'std']).T)
                print("=======================================================================")


# In[]:
# ================================基础操作 结束==============================


# In[]:
# ================================数据分布 开始==============================
# In[]:
# --------------------------------分类模型------------------------------
# 分类模型 ： 连续特征 数据分布（直方图、盒须图）： （不能有缺失值）
# f, axes = plt.subplots(2,2, figsize=(20, 18))
def class_data_distribution(data, feature, label, axes):
    if type(feature) == list:
        raise Exception('feature Type is Error, must not list')

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


# 分类模型 ： 2个连续特征的散点分布（因变量Y作为颜色区分）：
def class_data_scatter(x_data, one_f_name, two_f_name, y_data, axes):
    axes.scatter(x_data.loc[:, one_f_name], x_data.loc[:, two_f_name], c=y_data, s=10, cmap="rainbow")  # 蓝色
    axes.set_xlabel(one_f_name)  # x轴标签
    axes.set_ylabel(two_f_name)  # y轴标签


# 分类模型 ： 连续特征 与 因变量Y 散点图： （类似于 盒须图）
def class_data_with_y_scatter(data, feature_name, y_name):
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    # https://blog.csdn.net/weixin_42398658/article/details/82960379
    sns.FacetGrid(data, hue=y_name, size=5).map(plt.scatter, feature_name, y_name).add_legend()


# In[]:
# --------------------------------连续模型------------------------------
# 连续模型 连续特征 与 连续因变量Y 散点分布：
# （从左至右逐渐稀疏的散点图 → 第一反应是对Y取对数 → 特征取对数）
'''
方差齐次性： 测试两个连续特征的均方差的最佳方法是图形方式。 
通过圆锥（在图形的一侧较小的色散，在相反侧的较大色散）或菱形（在分布中心的大量点）来表示偏离均等色散的形状。
'''


# 例子： Pedro_Marcelino.py
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
def con_data_scatter(x_data, featur_name, y_data, y_name):
    f, axes = plt.subplots(2, 1, figsize=(15, 15))

    # 两种散点图绘制方式：
    axes[0].scatter(x_data[featur_name], y_data[y_name], c='#0000FF', s=10, cmap="rainbow")  # 蓝色
    axes[0].set_xlabel(featur_name)  # x轴标签
    axes[0].set_ylabel(y_name)  # y轴标签

    sns.regplot(x_data[featur_name], y_data[y_name], scatter=True, fit_reg=True, ax=axes[1])  # 加了趋势线


# 连续模型 ： 分类特征 与 连续因变量Y 四分位图 （盒须图）
# x轴为分类变量X， y轴为连续因变量Y 的盒须图 可以作为 斯皮尔曼相关系数 辅助可视化分析： 呈现逐 分类特征类别（序数） 递增，斯皮尔曼相关系数很高，分类特征X 对 连续因变量Y 有用
# 1、设置 分类特征为category类型，并手动设置category类别顺序
# 例子： Pedro_Marcelino.py
# ft.box_diagram(dat0, 'dist', 'price', axes, set_category=True, categories_=["石景山","丰台","朝阳","海淀","东城","西城"])
def box_diagram(data, x_axis_name, y_axis_name, axes=None, ymin=None, ymax=None, set_category=False, categories_=None,
                is_violin=False):
    if ymin is not None and ymax is not None:
        axes.axis(ymin=ymin, ymax=ymax)

    # 设置分类变量的字段类型为：category，并指定类别顺序，盒须图中按指定类别顺序显示
    if set_category and categories_ is not None and type(categories_) == list:
        # 参看 Python笔记： “赋值与地址”
        # 入参data → 调用变量train  改变指向  入参data → 新变量
        # 操作 入参data → 操作 新变量（不会影响 调用变量train）
        data = data[[x_axis_name, y_axis_name]]  # 即使显示的取所有列，也是新地址
        set_col_category(data, x_axis_name, categories_=categories_)

    if axes is None:
        f, axes = plt.subplots(1, 1, figsize=(10, 8))

    if is_violin:  # 小提琴图
        sns.violinplot(x=x_axis_name, y=y_axis_name, data=data, ax=axes)
    else:
        sns.boxplot(x=x_axis_name, y=y_axis_name, data=data, ax=axes)


# 2、自动将 分类特征的类型设置为category类型， 如果分类特征有np.nan值，则为分类特征的category类型添加一个category类别：missing
# 没有设置category类型顺序，后面再修改吧。 代码在： 1_EDA.py
def box_diagram_auto_col_category(data, categorical_features, y_name, function_type=1, is_violin=False):
    if type(categorical_features) is not list:
        raise Exception('categorical_features Type is Error, must list')

    all_feature = categorical_features.copy()
    all_feature.append(y_name)
    # 参看 Python笔记： “赋值与地址”
    # 入参data → 调用变量train  改变指向  入参data → 新变量
    # 操作 入参data → 操作 新变量（不会影响 调用变量train）
    data = data[all_feature]  # 即使显示的取所有列，也是新地址
    add_col_category(data, categorical_features)

    # 1、类别特征 盒须图/小提琴图 可视化
    # 确定这样传参？ 外层入参 直接传递给 内层函数？ 不经过内层函数的调用传参给内层函数？
    def boxplot(x, y, is_violin=is_violin, **kwargs):
        if is_violin:  # 小提琴图
            sns.violinplot(x=x, y=y)
        else:
            sns.boxplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    # 2、类别特征的柱形图可视化
    def bar_plot(x, y, **kwargs):
        sns.barplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    # 3、类别特征的每个类别频数可视化
    def count_plot(x, y, **kwargs):  # 这个 入参y 是参数占位符的意思，暂时不知道怎么修改
        sns.countplot(x=x)
        x = plt.xticks(rotation=90)

    if function_type == 1:
        statistics_function = boxplot
    elif function_type == 2:
        statistics_function = bar_plot
    elif function_type == 3:
        statistics_function = count_plot
    else:
        raise Exception('function_type Type is Error')

    f = pd.melt(data, id_vars=[y_name], value_vars=categorical_features)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(statistics_function, "value", y_name)


# 线性回归的可视化： sns.lmplot函数，了解下。


# In[]:
# --------------------------------复用------------------------------
# 连续特征 计算盒须图 区间：
def box_whisker_diagram_Interval(series, box_scale=1.5):
    if type(series) is not pd.Series:
        raise Exception('series Type is Error, must Series')

    max_val = np.max(series)
    min_val = np.min(series)
    #    mean_val = np.mean(series)
    iqr = series.quantile(0.75) - series.quantile(0.25)
    twenty_five_percent = series.quantile(0.25)
    five_percent = series.quantile(0.50)
    seventy_five_percent = series.quantile(0.75)
    upper_point = series.quantile(0.75) + box_scale * iqr
    down_point = series.quantile(0.25) - box_scale * iqr
    val_list = [min_val, down_point, twenty_five_percent, five_percent, seventy_five_percent, upper_point, max_val]
    return val_list


# 连续/分类模型 ： 连续 特征/因变量 直方图分布、盒须图： （不能有缺失值）
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


# f, axes = plt.subplots(1,2, figsize=(23, 8))
# feature为连续值（特征/因变量）
def con_data_distribution(data, feature, axes, fit_type=1, box_scale=1.5):
    if type(feature) == list:
        # 盒须图 要求 特征必须为单特征，不能传['x']进来
        raise Exception('feature Type is Error, must not list')

    data = get_notMissing_values(data, feature)

    distplot_title = 'Normal'
    if fit_type == 1:
        fit_function = scipy.stats.norm  # 正太分布
    elif fit_type == 2:
        fit_function = scipy.stats.lognorm  # 取log正太分布
        distplot_title = 'Log Normal'
    else:
        fit_function = scipy.stats.johnsonsu  # 无界约翰逊分布
        distplot_title = 'Johnson SU'

    sns.set()  # 切换到seaborn的默认运行配置
    # sns.distplot直方图（默认带KDE）
    sns.distplot(data[feature], bins=100, fit=fit_function, color='green', ax=axes[0])  # rug=True分布观测条显示

    # Get the fitted parameters used by the function
    # 现在只能以 norm正太分布方式 计算均值 和 标准差（其他两个分布，暂时没有查询到怎么计算）
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
    hist_left_more_index = data.loc[data[feature] < mu - 5 * sigma, feature].index
    hist_right_more_index = data.loc[data[feature] > mu + 5 * sigma, feature].index

    if len(hist_left_more_index) > 0:
        axes[0].plot((mu - 5 * sigma, mu - 5 * sigma), (0, 1), c='y', lw=1.5, ls='--', alpha=0.3)  # 5倍标准差之外的数据
    if len(hist_right_more_index) > 0:
        axes[0].plot((mu + 5 * sigma, mu + 5 * sigma), (0, 1), c='y', lw=1.5, ls='--', alpha=0.3)
    # Now plot the distribution
    axes[0].legend(['Normal dist. ( $\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
    axes[0].set_title(distplot_title + ' feature: ' + str(feature))
    axes[0].set_xlabel('')

    # 盒须图
    sns.boxplot(y=feature, data=data, ax=axes[1])
    iqr = data[feature].quantile(0.75) - data[feature].quantile(0.25)

    # 利用 众数 减去 中位数 的差值  除以  四分位距 来 查找是否有可能存在异常值（案例2：精准营销的两阶段预测模型3 的 9分钟）
    # 异常值一般表现为众数； 正常值一般表现为中位数； 看两者之间的差值 / 四分位距
    temp_outlier = abs((data[feature].mode().iloc[0,] - data[feature].median()) / iqr)
    print("如果 众数 减去 中位数 的差值 除以 四分位距 的值 %s 很大，需要进一步用直方图观测，对嫌疑大的变量进行可视化分析" % temp_outlier)

    upper_point = data[feature].quantile(0.75) + box_scale * iqr
    down_point = data[feature].quantile(0.25) - box_scale * iqr
    upper_more = data[data[feature] >= upper_point][feature]
    down_more = data[data[feature] <= down_point][feature]
    #    print(len(upper_more))
    #    print(len(down_more))

    if len(upper_more) != 0:
        axes[1].plot((-1, 1), (upper_point, upper_point), c='r', lw=1.5, ls='--', alpha=0.3)  # 99%
        print("Description of data larger than the upper bound is:")
        print(upper_more.describe())
    elif len(down_more) != 0:
        axes[1].plot((-1, 1), (down_point, down_point), c='b', lw=1.5, ls='--', alpha=0.3)  # 99%
        print("Description of data less than the lower bound is:")
        print(down_more.describe())
    axes[1].set_title('feature: ' + str(feature))
    axes[1].set_ylabel('')

    return (upper_more.index, down_more.index), (hist_left_more_index, hist_right_more_index)


# 连续/分类模型 ： 连续 特征/因变量 简易 直方图分布（快速查看）
def simple_con_data_distribution(data, numeric_features):
    f = pd.melt(data, value_vars=numeric_features)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")


# 原生 连续特征的直方图，不带KDE
def con_data_distribution_hist(data, feature=None, axe=None, color='blue'):
    if axe is None:
        fig, axe = plt.subplots(1, 1, figsize=(10, 8))

    if type(data) is pd.core.series.Series and feature is None:
        axe.hist(data, orientation='vertical', histtype='bar', color=color)
    elif type(data) is pd.core.frame.DataFrame and feature is not None:
        axe.hist(data[feature], orientation='vertical', histtype='bar', color=color)
    else:
        raise Exception('Type is Error')

    # 多连续特征线性关系图： （类似于皮尔森相似度的热力图） 在使用 皮尔森相似度 选择特征时也用到sns.pairplot画图函数


# 是 函数class_data_scatter 和 class_data_with_y_scatter的综合
# 很不错的文章： https://www.jianshu.com/p/6e18d21a4cad
# 1、连续模型： data中 连续特征 且 必须包含 连续因变量Y， hue=连续因变量Y，速度太慢无法运行
# 2、连续模型/分类模型： data中只包含连续特征（当然也可以包含 连续因变量Y），可以出结果
def multi_feature_linear_diagram(data, numeric_features, y_name=None, size=2):
    sns.set()
    if y_name is not None:
        sns.pairplot(data[numeric_features], hue=y_name, size=size)
    else:
        sns.pairplot(data[numeric_features], size=2, kind='scatter', diag_kind='kde')


# 分类特征 与 连续特征 柱状图（这里只是个示例，使用时重新写）
def barplot(axis_x, axis_y, p=sns.color_palette(), xlabel=u'用户职业', ylabel=u'逾期用户比例', label='train'):
    fig = plt.figure(figsize=(20, 20))

    ax1 = fig.add_subplot(3, 2, 1)
    ax1 = sns.barplot(axis_x, axis_y, alpha=0.8, color=p[0], label=label)
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)


# In[]:
# -----------------------------正太、偏度检测 开始-------------------------------
# In[]:
# 正太分布检测：
'''
原假设：样本来自一个正态分布的总体。
备选假设：样本不来自一个正态分布的总体。
w和p同向： w值越小； p-值越小、接近于0； 拒绝原假设。 (w值 与 偏度值skew 相反)
注意： 特征不能包含np.nan
'''


def normal_distribution_test(data, axe=None):
    var = data.columns
    shapiro_var = {}
    for i in var:
        # 参看 Python笔记： “赋值与地址”
        # 入参data → 调用变量train  改变指向  入参data → 新变量
        # 操作 入参data → 操作 新变量（不会影响 调用变量train）
        data = get_notMissing_values(data, i)
        shapiro_var[i] = scipy.stats.shapiro(data[i])  # 返回 w值 和 p值

    # 0列为w值； 1为p值。
    shapiro = pd.DataFrame(shapiro_var).T.sort_values(by=0, ascending=False)

    if axe is None:
        fig, axe = plt.subplots(1, 1, figsize=(15, 10))

    # bar的X轴顺序问题： https://blog.csdn.net/qq_35318838/article/details/80198307
    axe.bar(np.arange(len(shapiro.index)), shapiro[0], width=.4)
    axe.set_xticks(np.arange(len(shapiro.index)))
    axe.set_xticklabels(shapiro.index)
    axe.set_title("Normal distribution for shapiro")
    # 在柱状图上添加数字标签
    for a, b in zip(np.arange(len(shapiro.index)), shapiro[0]):
        # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
        axe.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)

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
skew、kurt说明参考 https://www.cnblogs.com/wyy1480/p/10474046.html
'''


def skew_distribution_test(data, axe=None, skew_limit=1):
    #    var = data.columns
    #    skew_var = {}
    #    for i in var:
    #        skew_var[i] = abs(data[i].skew())
    #    skew = pd.Series(skew_var).sort_values(ascending=False)

    skew = np.abs(data.skew()).sort_values(ascending=False)
    # 下面这种计算方式 在无缺失值情况下 和 上述2种计算方式 有小数点后两位的 差异。 有缺失值情况 待进一步验证。
    #    skew = data.apply(lambda x: np.abs(skew(x.dropna()))).sort_values(ascending=False)
    var_x_ln = skew.index[skew > skew_limit]  # skew的索引 --- data的列名

    kurt = np.abs(data.kurt()).sort_values(ascending=False)

    if axe is None:
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

    return skew, kurt, var_x_ln


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


# 上述两个方法的合并：
def normal_comprehensive(data, skew_limit=1):  # skew_limit=0.75
    if type(data) == pd.core.series.Series:
        temp_data = pd.DataFrame(data.copy())
    else:
        temp_data = data.copy()

    normal_distribution_test(temp_data)
    skew, kurt, var_x_ln = skew_distribution_test(temp_data, skew_limit)

    print(var_x_ln, len(var_x_ln))
    for i, var in enumerate(var_x_ln):
        f, axes = plt.subplots(1, 2, figsize=(23, 8))
        con_data_distribution(temp_data, var, axes)

    # 将偏度大于 阈值（默认1） 的连续变量 取对数
    if len(var_x_ln) > 0:
        logarithm_nagative(temp_data, var_x_ln, 2)

        normal_distribution_test(temp_data)
        skew, kurt = skew_distribution_test(temp_data, skew_limit)

        var_x_ln = skew.index[skew > skew_limit]  # skew的索引 --- data的列名
        print(var_x_ln, len(var_x_ln))
        for i, var in enumerate(var_x_ln):
            f, axes = plt.subplots(1, 2, figsize=(23, 8))
            con_data_distribution(temp_data, var, axes)


# Q-Q图检测 正太分布 （检测： 1、单特征正太分布； 2、扰动项ε/残差 正态分布）
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
# -----------------------------正太、偏度检测 结束-------------------------------

# In[]:
# -----------------------------正太、偏度处理 开始-------------------------------
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
    return X_rep  # 直接在原DataFrame上操作


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
# -----------------------------正太、偏度处理 结束-------------------------------
# In[]:
# ================================数据分布 结束==============================


# In[]:
# ================================特征选择 开始==============================
# In[]:
# 相似度计算1
# 皮尔森相似度
'''
1、皮尔森相似度（带 因变量Y） 先选出对Y 有贡献的： corrFunction_withY
2、再看 特征共线性： corrFunction
3、最后再对 选出的特征 与 Y 做pairplot图： feature_scatterplotWith_y
'''


# 1、特征选择：（data带 因变量Y）
def corrFunction_withY(data, label, is_show=True, image_width=20, image_hight=18):  # label： 因变量Y名称
    corr = data.corr()  # 计算各变量的相关性系数
    # 皮尔森相似度 绝对值 排序
    df_all_corr_abs = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr_abs.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                           inplace=True)
    temp_corr_abs = df_all_corr_abs[(df_all_corr_abs["Feature_1"] != label) & (df_all_corr_abs['Feature_2'] == label)]
    print(temp_corr_abs)
    print()
    # 皮尔森相似度 排序
    df_all_corr = corr.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation_Coefficient'},
                       inplace=True)
    temp_corr = df_all_corr[(df_all_corr["Feature_1"] != label) & (df_all_corr['Feature_2'] == label)]
    print(temp_corr)

    if is_show:
        temp_x = []
        for i, fe in enumerate(list(corr.index)):  # 也可以是： data.columns
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

    return temp_corr_abs, temp_corr


# 2、特征选择：特征共线性（data不带 因变量Y； 还可以做 方差膨胀系数）
def corrFunction(data, is_show=True, image_width=20, image_hight=18):
    '''
    1、特征间共线性：两个或多个特征包含了相似的信息，期间存在强烈的相关关系
    2、常用判断标准：两个或两个以上的特征间的相关性系数高于0.8
    3、共线性的影响：
    3.1、降低运算效率
    3.2、降低一些模型的稳定性
    3.3、弱化一些模型的预测能力

    取 [::2] 已经没有 交叉值对 的情况了：
    列1   列2
    A  →  B
    B  →  A
    '''
    # 建立共线性表格（是检测特征共线性的，所以排除Y）
    correlation_table = data.corr()
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

    if is_show:
        # 热力图
        temp_x = []
        for i, fe in enumerate(data.columns):
            temp_x.append("x" + str(i))
        xticks = temp_x  # x轴标签
        yticks = list(correlation_table.index)  # y轴标签
        fig = plt.figure(figsize=(image_width, image_hight))
        ax1 = fig.add_subplot(1, 1, 1)
        sns.heatmap(correlation_table, annot=True, cmap='rainbow', ax=ax1,
                    annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'})  #
        ax1.set_xticklabels(xticks, rotation=0, fontsize=14)
        ax1.set_yticklabels(yticks, rotation=0, fontsize=14)
        plt.show()

    return temp_corr_abs, temp_corr


# 3、data带 因变量Y  与  data不带 因变量Y  的综合 （入参data中带Y） 注意：代码还有缺陷
# 代码： juliencs.py
def corrFunction_all(data, y_name, f_importance_num=20, f_collinear_num=0.7):
    if f_importance_num > data.shape[1]:
        f_importance_num = data.shape[1]

    # 1、特征选择：（带 因变量Y）
    df_all_corr_abs, df_all_corr = corrFunction_withY(data, y_name, False)
    Feature_1_cols = df_all_corr_abs[0:f_importance_num]["Feature_1"]
    # 2、特征选择：特征共线性（data不带 因变量Y； 还可以做 方差膨胀系数） 返回值已经没有 交叉值对 情况
    temp_corr_abs, temp_corr = corrFunction(data.drop(y_name, axis=1), False)
    Feature_all = temp_corr_abs[
        temp_corr_abs.apply(lambda x: (x[0] in Feature_1_cols.values) & (x[1] in Feature_1_cols.values), axis=1)]
    Feature_all = Feature_all[Feature_all["Correlation_Coefficient"] >= 0.5]
    if Feature_all.shape[0] == 0:
        print("All Feature corr < 0.5")
        return np.nan, np.nan

    # 3、合并 特征共线性表格 和 特征重要性表格
    # 3.1、方式一： map方式
    '''
    fpd = pd.DataFrame(Feature_1_cols)
    fpd = fpd.set_index("Feature_1")
    fpd["sort"] = np.arange(fpd.shape[0], 0, -1)
    Feature_all_temp = Feature_all["Feature_1"].map(fpd["sort"])
    Feature_all_temp.name = "Feature_1_sort"
    Feature_all = consolidated_data_col(Feature_all, Feature_all_temp)
    '''
    # 3.2、方式二： merge表连接方式
    fpd = pd.DataFrame(Feature_1_cols)
    fpd["Feature_1_sort"] = np.arange(fpd.shape[0], 0, -1)
    # 索引自动重置
    # Feature_all = Feature_all.merge(fpd, left_on='Feature_1',right_on='Feature_1')
    Feature_all = Feature_all.merge(fpd, on=['Feature_1'])

    # 4、没有针对 Feature_1字段 进行 列顺序交换、合并等操作 的表格
    f_all_no_f1_merged = data_sort(Feature_all, ["Feature_1_sort", "Correlation_Coefficient"], [False, False])
    f_all_no_f1_merged = f_all_no_f1_merged[f_all_no_f1_merged["Correlation_Coefficient"] >= f_collinear_num]

    # 5、针对 Feature_1字段 进行 列顺序交换、合并等操作 的表格
    result = pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation_Coefficient", "Feature_1_sort"])
    # 按照 自变量 与 因变量Y 的相关度，从大到小顺序 进行循环（先保证 与Y高相关度的特征 在 特征共线性表格中Feature_1字段的 完整性）
    for i in Feature_1_cols:
        '''
        temp_corr_abs()函数返回值中已经没有 交叉值对，但本处的查询合并又会产生 交叉值对。
        1、 GrLivArea特征： 
        查询 Feature_1 得到的 temp_a： 
        88 GrLivArea AllFlrsSF 0.9954098084600875
        2、AllFlrsSF特征：
        查询 Feature_2 得到的 temp_b： 
        88 GrLivArea AllFlrsSF 0.9954098084600875
        3、也就是说： A、B特征相关，则A的Feature_1查询temp_a 和 B的Feature_2查询temp_b 一定会查询到同一行（索引相同）；
        而temp_b又执行 换列顺序 的操作，所以又出现了 交叉值对（索引相同，其实本就是同一行）。 
        这里 索引相同、其实是同一行的 交叉值对 和 X.corr函数生成的 交叉值对 值是一样的。所以后续需要过滤/删除掉 索引相同行（交叉值对）。
        4、temp_b执行 换列顺序 的操作，且又将temp_b[Feature_1_sort]字段值，赋值为temp_a[Feature_1_sort]。
        那么最终 过滤/删除掉 索引相同行（交叉值对） 之后，Feature_1、Feature_1_sort字段中 与Y低相关度的特征 会消失，
        因为 从大到小顺序进行循环 优先保证 与Y高相关度的特征 在 特征共线性表格中Feature_1字段的 完整性。
        '''
        temp_a = Feature_all[Feature_all["Feature_1"] == i]
        temp_val = temp_a["Feature_1_sort"].iloc[0]
        #    temp_list = ft.set_diff(result.index, temp_a.index)  # 1.1处： 在联和添加时 过滤掉 索引相同（其实是同一行）
        #    temp_a = temp_a.loc[temp_a.index.drop(temp_list[0])]
        result = consolidated_data_col(result, temp_a, axis=0)

        temp_b = Feature_all[Feature_all["Feature_2"] == i]
        #    temp_list = ft.set_diff(result.index, temp_b.index)  # 1.2处： 在联和添加时 过滤掉 索引相同（其实是同一行）
        #    temp_b = temp_b.loc[temp_b.index.drop(temp_list[0])]
        temp_b[["Feature_1", "Feature_2"]] = temp_b[["Feature_2", "Feature_1"]]
        temp_b["Feature_1_sort"] = temp_val
        result = consolidated_data_col(result, temp_b, axis=0)

    set_classif_col(result, "Feature_1_sort")  # 设置Feature_1_sort字段为int类型

    # 5.1、 删除重复索引（其实是同一行）
    result = duplicate_index(result)  # 2处： 一并删除重复索引： 索引相同（其实是同一行）； 和 1.1、1.2处 的作用相同（实现原理是相同的）。

    # 6、最终表格：
    # 6.1、直接排序：
    # 外层排序特征Feature_1_sort在前（优先），分组排序特征Correlation_Coefficient在后（次之），直接进行排序 实现分组排序。
    # 但前提是 必须自定义 外层排序特征Feature_1_sort。
    result_final = data_sort(result, ["Feature_1_sort", "Correlation_Coefficient"], [False, False])
    # 6.2、分组排序：（在我的需求 外层特征重要性需要排序时， 分组排序没有意义）
    '''
    使用分组排序：外层排序特征Feature_1_sort没用。 因为apply函数是按每个分组标签划分之后，再按该组内的特征进行排序，控制不了分组标签排序。
    且 每个分组标签 对应的 外层排序特征Feature_1_sort 都相同，没有意义。 但奇怪的是单独使用Feature_1_sort排序时，会带动其他数值类型特征进行排序...
    result_final = tc.groupby_apply_sort(result, ["Feature_1"], ["Correlation_Coefficient"], [False], group_keys=False)
    这样实现 分组排序groupby.apply 就没有意义。
    result_final = result_final.sort_values(by=["Feature_1_sort", "Correlation_Coefficient"], ascending=[False, False])
    '''

    # 7、保留 特征共线性 >= f_collinear_num 的数据（默认0.7）
    result_final = result_final[result_final["Correlation_Coefficient"] >= f_collinear_num]

    return f_all_no_f1_merged, result_final  # 两者维度相同； 数据结构不同


# 3.1 改进版本： （1、data带 因变量Y； 2、numeric_features不包含 因变量Y名称）
# 1、连续特征之间的 皮尔森相似度从大到小排序；  2、连续特征 与 连续因变量Y之间的 皮尔森相似度从大到小排序。
def feature_select_corr_withY(data, numeric_features, y_name, threshold=0.8):
    if type(numeric_features) != list:
        # 盒须图 要求 特征必须为单特征，不能传['x']进来
        raise Exception('numeric_features Type is Error, must list')
    if y_name in numeric_features:
        raise Exception("y_name can't in numeric_features")

    numeric_features = numeric_features.copy()
    numeric_features.append(y_name)
    temp_corr_abs_withY, temp_corr_withY = corrFunction_withY(data[numeric_features], y_name)
    numeric_features.remove(y_name)
    temp_corr_abs, temp_corr = corrFunction(data[numeric_features])

    # 连续特征之间的 皮尔森相似度 >= 0.8 要做筛选： 留下 一对连续特征之中 对 连续因变量Y 皮尔森相似度贡献大的连续特征
    temp_corr_abs = temp_corr_abs[temp_corr_abs['Correlation_Coefficient'] >= threshold]
    recovery_index(temp_corr_abs)
    del_list = list()
    equal_list = list()

    # 循环 连续特征之间的 皮尔森相似度 >= 0.8 的DataFrame
    for i in range(len(temp_corr_abs)):
        temp = temp_corr_abs.loc[i]
        # 如果 连续特征已经在 删除列表del_list 或 相等列表equal_list中， 则跳过
        if (temp['Feature_1'] in del_list) or (temp['Feature_2'] in del_list):
            continue;

        # 取出 该 连续特征 对 连续因变量Y 皮尔森相似度
        temp_withY1 = temp_corr_abs_withY[temp_corr_abs_withY['Feature_1'] == temp['Feature_1']]
        temp_withY2 = temp_corr_abs_withY[temp_corr_abs_withY['Feature_1'] == temp['Feature_2']]

        # 看 该对 连续特征当中， 哪个连续特征 对 连续因变量Y 皮尔森相似度 贡献小， 放入删除列表del_list
        if temp_withY1.iloc[0]['Correlation_Coefficient'] > temp_withY2.iloc[0]['Correlation_Coefficient']:
            del_list.append(temp_withY2.iloc[0]['Feature_1'])
        elif temp_withY1.iloc[0]['Correlation_Coefficient'] < temp_withY2.iloc[0]['Correlation_Coefficient']:
            del_list.append(temp_withY1.iloc[0]['Feature_1'])
        else:
            if i % 2:
                del_list.append(temp_withY1.iloc[0]['Feature_1'])
            else:
                del_list.append(temp_withY2.iloc[0]['Feature_1'])
            # 如果 该对 连续特征 对 连续因变量Y 皮尔森相似度 贡献相等， 相等列表equal_list中
            equal_list.append(temp_withY1.iloc[0]['Feature_1'] + '=' + temp_withY2.iloc[0]['Feature_1'])

    return del_list, equal_list


# 4、最后再对 选出的特征 与 Y 做pairplot图 （入参data中带Y）
# 4.1、连续特征之间 皮尔森相似度越大 则越趋近于 一条斜直线： 肯定要删除其中一个特征（哪个连续特征 对 连续因变量Y 皮尔森相似度小 删除）
# 4.2、连续特征 与 连续因变量Y 的皮尔森相似度 越大越好（趋近于一条斜直线）
def feature_scatterplotWith_y(data, cols):
    if type(cols) != list:
        # 盒须图 要求 特征必须为单特征，不能传['x']进来
        raise Exception('cols Type is Error, must list')
    sns.set()
    sns.pairplot(data[cols], size=2, kind='scatter', diag_kind='kde')
    plt.show();


# 特征选择：（特征 与 因变量Y） 皮尔森
def feature_corrWith_y(X, y_series, top_num=20):
    if type(X) is pd.core.series.Series:
        X = pd.DataFrame(X)
    return np.abs(X.corrwith(y_series)).sort_values(ascending=False)[:top_num]


# 相似度计算2
# 斯皮尔曼相似度：
'''
Spearman 秩次相关
Spearman 相关评估 两个连续或顺序变量 之间的单调关系。在单调关系中，变量倾向于同时变化，但不一定以恒定的速率变化。Spearman 相关系数基于每个变量的秩值（而非原始数据）。
Spearman 相关通常用于评估与顺序变量相关的关系。例如，您可能会使用 Spearman 相关来评估员工完成检验练习的顺序是否与他们工作的月数相关。
最好始终用散点图来检查变量之间的关系。相关系数仅度量线性 (Pearson) 或单调 (Spearman) 关系。也有可能存在其他关系。

理解： 
1、Spearman更多的应该是用于 顺序变量（分类变量中的 序数：类别有大小区分，如：矿石等级1,2,3）。 而连续变量的判断应该使用皮尔森相似度更准确。

2、Spearman运用于分箱： 分类模型（二分类）
2.1、连续变量X进行分箱 → 分类变量 → 每个小分箱求均值 → Spearman根据每个小分箱的均值计算得到pi： 顺序变量（分类变量中的 序数） → di = pi-qi
2.2、二分类变量Y → 每个小分箱求均值 → Spearman根据每个小分箱的均值（Y的1类别均值）计算得到qi： 顺序变量（分类变量中的 序数） → di = pi-qi
原始代码在： 3_Scorecard_model_case_1480.py

3、Spearman运用于检测 分类特征X（序数） 对 连续因变量Y 贡献度： 连续模型
x轴为分类变量X，y轴为连续因变量Y 的盒须图 可以作为 斯皮尔曼相关系数 辅助可视化分析： 呈现逐 分类特征类别（序数） 递增，斯皮尔曼相关系数很高，分类特征X 对 连续因变量Y 有用
3.1、分类变量X 需是 整数格式的 顺序变量（分类变量中的 序数：类别有大小区分），不能是普通的分类变量（分类变量中的 标称）
3.2、连续因变量Y

注意看： 3_Scorecard_model_case_1480.py 中Spearman相关代码和详细解释
'''


def feature_spearmanrWith_y(X_series, y_series):
    # 方式一： X_series 和 y_series 必须没有缺失值np.nan，否则即使失败
    r, p = scipy.stats.spearmanr(X_series, y_series)  # 返回两个值： r是斯皮尔曼值， p是概率值

    # 方式二： 允许X_series 或 y_series有缺失值np.nan，会自动剔除np.nan再进行计算
    #    r = X_series.corr(y_series, method='spearman') # 只返回： r是斯皮尔曼值
    return r, p


# 特征选择： 线性回归模型： 特征系数重要性 横向柱状图
def linear_regression_coef(model, X_train, axe, head_num=10, tail_num=10):
    coefs = pd.Series(model.coef_, index=X_train.columns)
    print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(head_num),
                           coefs.sort_values().tail(tail_num)])

    imp_coefs.plot(kind="barh", axes=axe)
    plt.title("Coefficients in the Ridge Model")
    plt.show()


'''
WOE转换 → 计算IV值 用法：
1、分类特征 直接 WOE转换 → 计算IV值 做 IV值 的 特征选择； 
2、特征分箱选择最优分箱区间：
2.1、连续特征 WOE转换 → 计算IV值 选择最优分箱区间；
2.2、分类特征“概化”（再分箱） WOE转换 → 计算IV值 选择最优分箱区间（只是理论上，代码未实现这种思路。实际代码：step1_donations_logit.py）
2.3、最后再做 IV值 的 特征选择

IV值 取值区间如下：
1、0 --- 0.02 弱
2、0.02 --- 0.1 有价值
3、0.1 --- 0.4 很有价值
4、0.4 --- 0.6 非常强
5、0.6 以上 单独将变量拿出来，如果是信用评级，单独做一条规则。
'''


# 1.1.1、分类模型（二分类）： 分类特征 与 二分类因变量Y 的相关性 （分类特征 直接求WOE → IV值）
# 注意： 分类模型（二分类）： 1、连续变量分箱 或 2、分类变量“概化”（再分箱） 都需要经过： WOE分箱→IV值检测 确定分箱区间； 工具类： Binning_tools.py
def category_feature_iv(data, category_feature_name, y_name):  # data 包含 二分类因变量Y
    bins_df = tc.groupby_value_counts_unstack(data, category_feature_name, y_name)
    df_change_colname(bins_df, columns={0: 'count_0', 1: 'count_1'})
    bins_df["total"] = bins_df.count_0 + bins_df.count_1  # 一个箱子当中所有的样本数： 按列相加
    bins_df["percentage"] = bins_df.total / bins_df.total.sum()  # 一个箱子里的样本数 占 所有样本 的比例
    bins_df["bad_rate"] = bins_df.count_1 / bins_df.total  # 一个箱子坏样本的数量 占 一个箱子里所有样本数的比例
    bins_df["good%"] = bins_df.count_0 / bins_df.count_0.sum()  # 一个箱子 好样本 的数量 占 所有箱子里 好样本 的比例
    bins_df["bad%"] = bins_df.count_1 / bins_df.count_1.sum()  # 一个箱子 坏样本 的数量 占 所有箱子里 坏样本 的比例
    bins_df["good_cumsum"] = bins_df["good%"].cumsum()
    bins_df["bad_cumsum"] = bins_df["bad%"].cumsum()
    bins_df["woe"] = np.log(bins_df["bad%"] / bins_df["good%"])

    rate = bins_df["bad%"] - bins_df["good%"]
    iv = np.sum(rate * bins_df.woe)
    print(iv)
    return bins_df, iv


# 1.1.2、所有 分类特征 的IV值
def get_all_category_feature_ivs(data, var_d, y_name, use_woe_library=False):
    from woe import WoE  # 大神封装的WOE工具类： woe.py

    if type(var_d) != list:
        raise Exception('var_d Type is Error, must list')

    iv_d = {}
    for i in var_d:  # 自变量X（分类）
        if use_woe_library:
            iv_d[i] = WoE(v_type='d').fit(data[i], data[y_name]).iv  # v_type='d'求分类变量的WOE→IV值
        else:
            _, iv_d[i] = category_feature_iv(data, i, y_name)

    iv_d = pd.Series(iv_d).sort_values(ascending=False)
    var_d_s = list(iv_d[iv_d > 0.02].index)  # 取IV值大于0.02的 分类特征
    return iv_d, var_d_s


# 1.1.3、将 WOE值 映射到 原始数据：
def all_feature_woe_mapping(data, var_d, y_name, use_woe_library=False):
    from woe import WoE  # 大神封装的WOE工具类： woe.py

    for i in var_d:
        if use_woe_library:
            data[i + "_woe"] = WoE(v_type='d').fit_transform(data[i], data[y_name])
        else:
            bins_df, _ = category_feature_iv(data, i, y_name)
            data[i + "_woe"] = data[i].map(bins_df['woe'])


# 1.2、分类模型（二分类）： 连续特征 与 二分类因变量Y 的相关性 （连续特征 分箱后 求WOE → IV值） 粗略的检测一下
# 注意： 分类模型（二分类）： 1、连续变量分箱 或 2、分类变量“概化”（再分箱） 都需要经过： WOE分箱→IV值检测 确定分箱区间； 工具类： Binning_tools.py
# 这里直接使用了 大神封装的WOE工具类： woe.py（直接指定qcut=3箱 简略求 连续特征 分箱后 求WOE → IV值，没细看源码），连续变量分箱 还是应该按照 Binning_tools.py 中的步骤进行。
def get_all_con_feature_ivs(data, var_c, y_name):
    from woe import WoE  # 大神封装的WOE工具类： woe.py

    if type(var_c) != list:
        raise Exception('var_c Type is Error, must list')

    iv_c = {}
    for i in var_c:  # 自变量X（连续）
        # qnt_num=3： pd.qcut为3箱；
        # v_type='c'： 特征类型（'c'：连续特征； 'd'：分类特征）， 默认 v_type='c'
        # t_type='b'： 因变量类型（'c'：连续因变量； 'b'：二分类因变量）， 默认 t_type='b'
        iv_c[i] = WoE(v_type='c', t_type='b', qnt_num=3).fit(data[i], data[y_name]).iv  # v_type='d'求分类变量的WOE→IV值

    iv_c = pd.Series(iv_c).sort_values(ascending=False)
    var_c_s = list(iv_c[iv_c > 0.02].index)  # 取IV值大于0.02的 连续特征
    return iv_c, var_c_s


'''
分箱 会造成信息丢失、精确度降低。但同时增加了模型的泛化能力。
疑问：
分类模型（二分类） 的 特征筛选：
1、连续特征 WOE分箱 → 选择IV值>0.02： 是使用 分箱后的连续特征； 还是继续使用 原始连续特征； 还是两者都保留？
2、分类特征“概化”（再分箱） WOE分箱 → 选择IV值>0.02（只是理论上，代码未实现这种思路）： 是使用 再分箱（“概化”）后的分类特征； 还是继续使用 原始分类特征； 还是两者都保留？

例子：
1、在 step1_donations_logit.py 中经过 IV值 的特征删选后， 依然使用： 
1.1、原始连续特征； 
1.2、分类特征“概化”（再分箱） 后的 分类特征（原分类特征删除）
1.3、就为了最后做： PCA主成分分析、因子分析等 需要 连续特征 的算法。

2、在 2_Scorecard_model_case_My.py 逻辑评分卡业务中 所有特征都需转换为WOE值： 
2.1、连续特征 WOE分箱 → IV值 选择最优分箱区间； 
2.2、计算 连续特征 按最优分箱区间分箱后 每个分箱区间的WOE值，最后完成连续特征的WOE值映射；
2.3、所有 原始特征数据 现在都映射成WOE值，进模型训练（具体看代码中的逻辑）
2.4、最后再做 IV值 的 特征选择
'''

# In[]:
# ================================特征选择 结束==============================


# In[]:
# ================================特征创建 开始==============================
# In[]:
'''
1、现有功能的简化
2、现有功能的组合
3、现有特征中的多项式
'''
# 1、现有功能的简化
# 1.1、分类特征： 将 分桶类别 进行压缩 ： 如可以调用 category_manual_coding 自定义函数

# 2、现有功能的组合

# 3、现有特征中的多项式


# In[]:
# ================================特征创建 结束==============================


# In[]:
# ================================线性回归特征分析 开始==============================
# In[]:
from sklearn.metrics import mean_squared_error as MSE  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

# 菜菜的 8_1_LinearRegression.py 更详细： 1、使用统计学方式； 2、改进线性回归模型 解决 多重共线性（岭回归、Lasso）

# 拟合优度
# R^2
# 库方式： 1、r2_score(Ytest, yhat_test) 2、reg.score(Xtest_,Ytest) 3、cross_val_score(reg, Xtest_, Ytest, cv=10, scoring="r2").mean()
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
网上查询后用下面的公式：
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
def heteroscedastic(X, Y, col_list):  # Y的列名必须为：'Y'
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


def heteroscedastic_singe(X, Y, col, y_name='Y', is_auto=True):
    temp_X = X[col]
    temp_Y = pd.DataFrame(Y)
    df_change_colname(temp_Y, {y_name: 'Y'})
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

    if is_auto:
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
    else:
        r_sq = {'Y~' + col: lm_s1.rsquared}

    return r_sq


# 1.2、扰动项ε 服从正太分布 （QQ检验）
# 代码在： 8_1_LinearRegression.py → ft.disturbance_term_normal(Xtrain, Ytrain, col_list)
def disturbance_term_normal(X, Y, col_list):
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


# temp_index = ft.studentized_residual(exp['Income_ln'], exp['avg_exp_ln'], ['Income_ln'], 'avg_exp_ln', num=2)
def studentized_residual(Xtrain, Ytrain, X_names, Y_name='Y', num=3):
    temp_data = pd.concat([Xtrain, Ytrain], axis=1)
    formula = Y_name + '~' + '+'.join(X_names)

    lm_s = ols(formula, data=temp_data).fit()
    print(lm_s.rsquared, lm_s.aic)
    temp_data['Pred'] = lm_s.predict(temp_data)
    temp_data['resid'] = lm_s.resid  # 残差随着x的增大呈现 喇叭口形状，出现异方差
    temp_data.plot('Pred', 'resid', kind='scatter')  # Pred = β*Income，随着预测值的增大，残差resid呈现 喇叭口形状

    temp_data['resid_t'] = (temp_data['resid'] - temp_data['resid'].mean()) / temp_data['resid'].std()

    temp_data2 = temp_data[abs(temp_data['resid_t']) <= num]
    lm_s2 = ols(formula, temp_data2).fit()
    print(lm_s2.rsquared, lm_s2.aic)
    temp_data2['Pred'] = lm_s2.predict(temp_data2)
    temp_data2['resid'] = lm_s2.resid
    temp_data2.plot('Pred', 'resid', kind='scatter')
    lm_s2.summary()

    return temp_data[abs(temp_data['resid_t']) > num].index  # 返回强影响点索引


# 2.2、强影响点分析 更多指标： statemodels包提供了更多强影响点判断指标 （太耗时，最好不要用了）
def strong_influence_point(Xtrain, Ytrain):
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
# 3、方差膨胀因子： 特征之间 互相进行线性回归预测
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
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    print(formula)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2), formula  # 结果和Sklearn包计算结果相同


# Sklearn版本（结果相同）
def vif_sklearn(df, col_i):
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    Xtrain = df[cols_noti]
    Ytrain = df[col_i]
    reg = LR().fit(Xtrain, Ytrain)
    yhat = reg.predict(Xtrain)
    r2 = r2_score_customize(Ytrain, yhat, 1)
    return 1. / (1. - r2), formula


# 先求没有取对数的方差膨胀因子，再求取了对数的方差膨胀因子 方便对比
def variance_expansion_coefficient(df, cols, types=1):
    temp_df = df[cols]  # DataFrame 即使显示的取所有列，也是新地址
    temp_dict = {}
    temp_dict_ln = {}

    for i in temp_df.columns:
        if types == 1:
            temp_v = vif(df=temp_df, col_i=i)
        else:
            temp_v = vif_sklearn(df=temp_df, col_i=i)
        temp_dict[temp_v[1]] = temp_v[0]
        print(i, '\t', temp_v[0])
        print()

    print("-" * 30)

    logarithm(temp_df, temp_df.columns)
    col_list_ln = [i + "_ln" for i in cols]
    temp_df = temp_df[col_list_ln]
    for i in temp_df.columns:
        if types == 1:
            temp_v = vif(df=temp_df, col_i=i)
        else:
            temp_v = vif_sklearn(df=temp_df, col_i=i)
        temp_dict_ln[temp_v[1]] = temp_v[0]
        print(i, '\t', temp_v[0])
        print()

    return temp_dict, temp_dict_ln


# 训练集 与 测试集 拟合：
# f, axes = plt.subplots(3,1, figsize=(23, 18))
def fitting_comparison(y_train_true, y_train_predict, y_test_true, y_test_predict, axe):
    x_axis = np.hstack((y_train_predict, y_test_predict))
    y_axis = np.hstack((y_train_true, y_test_true))

    axe[0].scatter(y_train_predict, y_train_true, c="blue", marker="s", label="Training data")
    axe[0].scatter(y_test_predict, y_test_true, c="lightgreen", marker="s", label="Validation data")
    axe[0].set_title("Linear regression")
    axe[0].set_xlabel("Predicted values")
    axe[0].set_ylabel("Real values")
    axe[0].legend(loc="upper left")
    axe[0].plot([np.min(x_axis).round(2), np.max(x_axis).round(2)], [np.min(y_axis).round(2), np.max(y_axis).round(2)],
                c="red")

    axe[1].plot(range(len(y_train_true)), sorted(y_train_true), c="darkblue", label="Train_Data")
    axe[1].plot(range(len(y_train_predict)), sorted(y_train_predict), c="blue", label="Train_Predict")
    axe[1].legend()
    axe[2].plot(range(len(y_test_true)), sorted(y_test_true), c="darkgreen", label="Test_Data")
    axe[2].plot(range(len(y_test_predict)), sorted(y_test_predict), c="lightgreen", label="Test_Predict")
    axe[2].legend()

    plt.show()


# 训练集 与 测试集 残差
# f, axes = plt.subplots(1,1, figsize=(18, 10))
def linear_model_residuals(y_train_true, y_train_predict, y_test_true, y_test_predict, axe):
    x_axis = np.hstack((y_train_predict, y_test_predict))

    axe.scatter(y_train_predict, y_train_predict - y_train_true, c="blue", marker="s", label="Training data")
    axe.scatter(y_test_predict, y_test_predict - y_test_true, c="lightgreen", marker="s", label="Validation data")
    axe.set_title("Linear regression")
    axe.set_xlabel("Predicted values")
    axe.set_ylabel("Residuals")
    axe.legend(loc="upper left")
    axe.hlines(y=0, xmin=np.min(x_axis).round(2), xmax=np.max(x_axis).round(2), color="red")
    plt.show()


# 线性回归模型： 预测值 与 真实值 的 散点分布： X轴为连续特征， Y轴为预测值/真实值
def feature_predic_actual_scatter(X, y, feature_name, y_name, model):
    f, axes = plt.subplots(2, 1, figsize=(15, 15))

    subsample_index = tc.get_randint(low=0, high=len(y), size=50)

    axes[0].scatter(X[feature_name][subsample_index], y[subsample_index], color='black')
    axes[0].scatter(X[feature_name][subsample_index], model.predict(X.loc[subsample_index]), color='blue')
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel(y_name)
    axes[0].legend(['True ' + y_name, 'Predicted ' + y_name], loc='upper right')
    print("The predicted " + y_name + " is obvious different from true " + y_name)

    # 连续特征 与 连续因变量Y（真实值） 散点分布：
    # （从左至右逐渐稀疏的散点图 → 第一反应是对Y取对数 → 特征取对数）
    sns.regplot(X[feature_name], y, scatter=True, fit_reg=True, ax=axes[1])  # 加了趋势线

    plt.show()


# 线性回归模型（线性回归、岭回归）： 方差与泛化误差 学习曲线：
'''
一个集成模型(f)在未知数据集(D)上的泛化误差 ，由方差(var)，偏差(bais)和噪声(ε)共同决定。其中偏差就是训练集上的拟合程度决定，方差是模型的稳定性决定，噪音(ε)是不可控的。而泛化误差越小，模型就越理想。
E(f; D) = bias^2 + var + ε^2
其中可控部分： bias^2 + var； 不可控部分： 噪音(ε)
'''


def linear_model_comparison(X, y, cv_customize=5, start=1, end=1001, step=100, linear_show=True):
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
        # 1 - R^2均值 = 偏差， 所以 简化起见 用 R^2均值 代表偏差（R^2均值越小，偏差越大； R^2均值越大，偏差越小）
        # 交叉验证 的 R^2均值： 不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 均值： 不同模型R^2拟合优度的准确性
        ridge_r2_score = ridge_score.mean()  # 简化起见 用 R^2均值 代表 偏差
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
    print("R^2起始正则化阈值%f:R^2起始值%f，R^2最大值正则化阈值%f:R^2最大值%f，R^2差值%f" % (start_Alpha, start_R2, maxR2_Alpha, maxR2, diff_R2))

    # 当R^2最大值时，求 R^2方差Var的最大值，用R^2的变化差值 与 R^2方差的变化差值 再进行比较
    start_R2VaR = ridge_r2var_scores[0]
    R2VarR_Index = alpharange.tolist().index(maxR2_Alpha)
    R2varR = ridge_r2var_scores[R2VarR_Index]
    diff_R2varR = R2varR - start_R2VaR
    print("R^2方差起始正则化阈值%f:R^2方差起始值%f，R^2最大值正则化阈值%f:R^2最大值对应方差值%f，R^2方差差值%f" % (
    alpharange[0], start_R2VaR, maxR2_Alpha, R2varR, diff_R2varR))
    print("R^2方差差值/R^2差值 = %f" % (diff_R2varR / diff_R2))

    print('-' * 30)

    # 1、打印R2最大值时对应的正则化参数取值； 2、并打印R2最大值； 3、并打印R^2最大值对应的R^2方差值
    #    print(alpharange[ridge_r2_scores.index(max(ridge_r2_scores))], max(ridge_r2_scores), ridge_r2var_scores[ridge_r2_scores.index(max(ridge_r2_scores))])
    print("R2最大值时对应的正则化参数取值:%f； R2最大值:%f； R^2最大值对应的R^2方差值:%f" % (
    alpharange[ridge_r2_scores.index(max(ridge_r2_scores))], max(ridge_r2_scores),
    ridge_r2var_scores[ridge_r2_scores.index(max(ridge_r2_scores))]))

    # 2、打印R2方差最小值时对应的正则化参数取值； 2、并打印R2方差最小值对应的R2值； 3、并打印R2方差最小值
    #    print(alpharange[ridge_r2var_scores.index(min(ridge_r2var_scores))], ridge_r2_scores[ridge_r2var_scores.index(min(ridge_r2var_scores))], min(ridge_r2var_scores))
    print("R2方差最小值时对应的正则化参数取值:%f； R2方差最小值对应的R2值:%f； R2方差最小值:%f" % (
    alpharange[ridge_r2var_scores.index(min(ridge_r2var_scores))],
    ridge_r2_scores[ridge_r2var_scores.index(min(ridge_r2var_scores))], min(ridge_r2var_scores)))

    # 3、打印泛化误差可控部分最小值时对应的正则化参数取值； 2、并打印泛化误差可控部分最小值时对应的R2值； 3、并打印泛化误差可控部分最小值时对应的R2方差值； 4、并打印泛化误差可控部分最小值
    #    print(alpharange[ridge_ge.index(min(ridge_ge))],ridge_r2_scores[ridge_ge.index(min(ridge_ge))],ridge_r2var_scores[ridge_ge.index(min(ridge_ge))], min(ridge_ge))
    print("泛化误差可控部分最小值时对应的正则化参数取值:%f； 泛化误差可控部分最小值时对应的R2值:%f； 泛化误差可控部分最小值时对应的R2方差值:%f； 泛化误差可控部分最小值:%f" % (
    alpharange[ridge_ge.index(min(ridge_ge))], ridge_r2_scores[ridge_ge.index(min(ridge_ge))],
    ridge_r2var_scores[ridge_ge.index(min(ridge_ge))], min(ridge_ge)))

    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2_scores, color="red", label="Ridge R2_Mean")
    if linear_show:
        plt.plot(alpharange, linear_r2_scores, color="orange", label="LR R2_Mean")
    plt.title(title_mean)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2var_scores, color="red", label="Ridge R2_Var")
    if linear_show:
        plt.plot(alpharange, linear_r2var_scores, color="orange", label="LR R2_Var")
    plt.title(title_var)
    plt.legend()
    plt.show()

    # R2_Var值非常小，从0.0038→0.0045平稳缓慢增加，所以看不出变宽的痕迹。
    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_r2_scores, c="k", label="R2_Mean")
    plt.plot(alpharange, ridge_r2_scores + np.array(ridge_r2var_scores), c="red", linestyle="--", label="R2_Var")
    plt.plot(alpharange, ridge_r2_scores - np.array(ridge_r2var_scores), c="red", linestyle="--")
    plt.legend()
    plt.title("Ridge R2_Mean vs R2_Var")
    plt.show()

    # 绘制 化误差的可控部分
    plt.figure(figsize=(10, 8))
    plt.plot(alpharange, ridge_ge, c="gray", linestyle='-.')
    plt.title("Ridge Generalization error")
    plt.show()


# In[]:
# ================================线性回归特征分析 结束==============================


# In[]:
# ====================================学习曲线 开始==================================
'''
L、学习曲线顺序： 基于样本量：如果过拟合（训练集、测试集相差过远） →  基于超参数：比较现实的 目标 是将训练集效果降低，从而避免过拟合 →  基于样本量：再次检测过拟合情况    
代码： 10_1_XGBoost.py 中 一、基于样本量(交叉验证学习曲线函数) → 二、基于超参数（按顺序 依次确定 超参数）     
'''


# In[]:
# -----------------------------1、基于样本量 开始-------------------------------

# 1、基于learning_curve样本量绘图函数（通用）： Sklearn模型通用的基于样本量的学习曲线（X轴： 训练集样本量； Y轴：训练集、测试集得分）
# estimator可以为： 1、LinearR()； 2、XGBR(n_estimators=100,random_state=420)
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
    print("验证集最大分数 对应于 交叉验证训练集阈值%d, 验证集最大分数%f" % (
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


# 2、于MSE绘制学习曲线（样本量）
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


# 3、基于R^2值绘制学习曲线（样本量）
def plot_learning_curve_r2_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(r2_score(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(r2_score(y_test, y_test_predict))

    plt.plot([i for i in range(1, len(X_train) + 1)], train_score, label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)], test_score, label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, -0.1, 1.1])
    plt.show()


# -----------------------------1、基于样本量 结束-------------------------------


# In[]:
# -----------------------------2、基于超参数 开始-------------------------------

# XGBoost必调超参数： 1、n_estimators； 2、gamma/γ复杂度惩罚项 或 max_depth树深度（选其一）

'''
L1、基于超参数学习曲线顺序： 确定n_estimators → 确定subsample → 确定learning_rate → 确定gamma （主要是理解 梯度提升树中这些超参数原理）
代码： 10_1_XGBoost.py 中 二、基于超参数（按顺序 依次确定 超参数）  

XGB中与梯度提升树的过程相关的四个参数：n_estimators，learning_rate ，silent，subsample。这四个参数的主要目的，其实并不是提升模型表现，更多是了解梯度提升树的原理。
现在来看，我们的梯度提升树可是说是由三个重要的部分组成：
1. 一个能够衡量集成算法效果的，能够被最优化的损失函数Obj。
2. 一个能够实现预测的弱评估器fk(x)
3. 一种能够让弱评估器集成的手段，包括我们讲解的：迭代方法，抽样手段，样本加权等等过程

XGBoost是在梯度提升树的这三个核心要素上运行，它重新定义了损失函数和弱评估器，并且对提升算法的集成手段进行了改进，实现了运算速度和模型效果的高度平衡。
并且，XGBoost将原本的梯度提升树拓展开来，让XGBoost不再是单纯的树的集成模型，也不只是单单的回归模型。
只要我们调节参数，我们可以选择任何我们希望集成的算法，以及任何我们希望实现的功能。
'''


# 1.1、SKLearn库的XGBoost：
def getModel(i, model_name, hparam_name, prev_hparam_value, random_state, silent=True):
    from xgboost import XGBRegressor as XGBR

    if model_name == "XGBR":
        if hparam_name == "n_estimators":
            reg = XGBR(n_estimators=i, random_state=random_state, silent=silent)  # 不打印训练过程记录
        elif hparam_name == "subsample" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value[0], subsample=i, random_state=random_state, silent=silent)
        elif hparam_name == "learning_rate" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value[0], subsample=prev_hparam_value[1], learning_rate=i,
                       random_state=random_state, silent=silent)
        elif hparam_name == "gamma" and prev_hparam_value is not None:
            reg = XGBR(n_estimators=prev_hparam_value[0], subsample=prev_hparam_value[1],
                       learning_rate=prev_hparam_value[2], gamma=i, random_state=random_state, silent=silent)
        else:
            raise RuntimeError('Hparam Error')
    return reg


# 1.2、方差与泛化误差 学习曲线： （参考 “线性回归模型（线性回归、岭回归）： 方差与泛化误差 学习曲线： linear_model_comparison”）
# 暂时只支持 SKLearn库的XGBoost。要支持其他模型，等待再开发 getModel函数。
'''
一个集成模型(f)在未知数据集(D)上的泛化误差 ，由方差(var)，偏差(bais)和噪声(ε)共同决定。其中偏差就是训练集上的拟合程度决定，方差是模型的稳定性决定，噪音(ε)是不可控的。而泛化误差越小，模型就越理想。
E(f; D) = bias^2 + var + ε^2
其中可控部分： bias^2 + var； 不可控部分： 噪音(ε)

注意： SKLearn库的XGBoost的gamma超参数在小范围区间容易波动，应使用xgb原生库。 但在稍大范围区间和xgb原生库差不多（实际中还是使用xgb原生库）
'''


def learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, model_name="XGBR", hparam_name="n_estimators",
                                prev_hparam_value=None, random_state=420):
    rs = []
    var = []
    ge = []
    for i in axisx:
        reg = getModel(i, model_name, hparam_name, prev_hparam_value, random_state)
        cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
        # 因 R^2=(-∞,1]， R^2拟合优度： 模型捕获到的信息量 占 真实标签中所带的信息量的比例
        # 1 - R^2均值 = 偏差， 所以 简化起见 用 R^2均值 代表偏差（R^2均值越小，偏差越大； R^2均值越大，偏差越小）
        # 交叉验证 的 R^2均值： 不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 均值： 不同模型R^2拟合优度的准确性
        rs.append(cvresult.mean())  # 简化起见 用 R^2均值 代表 偏差
        # 方差：交叉验证 的 R^2方差：不同训练集训练出多个模型 分别预测不同测试集得到多个预测值集合 --- 多个R^2拟合优度， 多个R^2拟合优度 的 方差： 不同模型R^2拟合优度的离散程度
        var.append(cvresult.var())  # R^2 方差
        # 计算泛化误差的可控部分
        ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())

    # 1、打印R2最大值时对应的 n_estimators/subsample/learning_rate/gamma 参数取值； 2、并打印R2最大值； 3、并打印R^2最大值对应的R^2方差值
    # print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
    print("R2最大值时对应的%s参数取值:%f； R2最大值:%f； R^2最大值对应的R^2方差值:%f" % (
    hparam_name, axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))]))

    # 2、打印R2方差最小值时对应的 n_estimators/subsample/learning_rate/gamma 参数取值； 2、并打印R2方差最小值对应的R2值； 3、并打印R2方差最小值
    # print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
    print("R2方差最小值时对应的%s参数取值:%f； R2方差最小值对应的R2值:%f； R2方差最小值:%f" % (
    hparam_name, axisx[var.index(min(var))], rs[var.index(min(var))], min(var)))

    # 3、打印泛化误差可控部分最小值时对应的 n_estimators/subsample/learning_rate/gamma 参数取值； 2、并打印泛化误差可控部分最小值时对应的R2值； 3、并打印泛化误差可控部分最小值时对应的R2方差值； 4、并打印泛化误差可控部分最小值
    #    print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
    print("泛化误差可控部分最小值时对应的%s参数取值:%f； 泛化误差可控部分最小值时对应的R2值:%f； 泛化误差可控部分最小值时对应的R2方差值:%f； 泛化误差可控部分最小值:%f" % (
    hparam_name, axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge)))

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


# SKLearn库的XGBoost： 由于 eta迭代次数（𝜂/步长/learning_rate） 和 n_estimators 超参数密切相关，需要一起搜索，所以使用GridSearchCV
# 代码： 10_1_XGBoost.py 中 二、基于超参数（按顺序 依次确定 超参数） → 3、eta（迭代决策树）
'''
注意： 所以通常，我们不调整eta： 𝜂/步长/learning_rate，即便调整，一般它也会在[0.01,0.2]之间变动。
如果我们希望模型的效果更好，更多的可能是从树本身的角度来说，对树进行剪枝，而不会寄希望于调整𝜂。（𝜂通常是用来调整运行时间的）
'''


def eta_and_n_estimators(Xtrain, Ytrain, Xtest, Ytest, cv=None):
    from xgboost import XGBRegressor as XGBR
    from sklearn.model_selection import GridSearchCV

    if cv is None:
        cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

    # 运行时间2分多钟
    param_grid = {
        'n_estimators': np.arange(100, 300, 10),
        'learning_rate': np.arange(0.05, 1, 0.05)
    }
    reg = XGBR(random_state=420)
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, verbose=1, cv=cv,
                               scoring='r2')  # neg_mean_squared_error
    grid_search.fit(Xtrain, Ytrain)
    '''
    reg:linear is now deprecated in favor of reg:squarederror.
    现在不推荐使用reg：linear，而推荐使用reg：squarederror。
    XGBoost的重要超参数objective损失函数选项： reg：linear → reg：squarederror
    '''

    print(grid_search.best_score_)  # 0.870023216964111
    print(grid_search.best_params_)  # {'learning_rate': 0.25, 'n_estimators': 260}
    best_reg = grid_search.best_estimator_  # 最佳分类器
    print(best_reg)
    testScore = best_reg.score(Xtest, Ytest)
    print("GridSearchCV测试结果：", testScore)  # 0.8996310370746 分数比 学习曲线的低。。。
    # [Parallel(n_jobs=1)]: Done 1900 out of 1900 | elapsed:  2.1min finished

    # 后使用neg_mean_squared_error评价指标，和R^2结果相同。


# ---------------------------------------------------------------------------


# XGB原生库： 方差与泛化误差 学习曲线： 自定义交叉验证（XGBoost原生库） 自己写的
# 现为 num_round/n_estimators已固定，画gamma/γ （循环其他超参数也行的， 和上面Sklearn库的 learning_curve_r2_customize 方差与泛化误差学习曲线是一样的意思）
# 入参 X、y 都是矩阵格式
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
        print(param_fixed)  # {'silent': True, 'obj': 'reg:linear', 'eval_metric': 'rmse', 'gamma': 10}

        # 虽然传进来的 评估指标是：rmse， 但并没有使用API得到rmse值，而是predict得到预测结果后，用r2和MSE得到评估结果。
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

    # 1、打印R2最大值时对应的 n_estimators/subsample/learning_rate/gamma 参数取值； 2、并打印R2最大值； 3、并打印R^2最大值对应的R^2方差值
    # print(axisx[rs_all_test.index(max(rs_all_test))], max(rs_all_test), var_all_test[rs_all_test.index(max(rs_all_test))])
    print("R2最大值时对应的%s参数取值:%f； R2最大值:%f； R^2最大值对应的R^2方差值:%f" % (
    axisx[rs_all_test.index(max(rs_all_test))], max(rs_all_test), var_all_test[rs_all_test.index(max(rs_all_test))]))

    # 2、打印R2方差最小值时对应的 n_estimators/subsample/learning_rate/gamma 参数取值； 2、并打印R2方差最小值对应的R2值； 3、并打印R2方差最小值
    # print(axisx[var_all_test.index(min(var_all_test))], rs_all_test[var_all_test.index(min(var_all_test))], min(var_all_test))
    print("R2方差最小值时对应的%s参数取值:%f； R2方差最小值对应的R2值:%f； R2方差最小值:%f" % (
    axisx[var_all_test.index(min(var_all_test))], rs_all_test[var_all_test.index(min(var_all_test))],
    min(var_all_test)))

    # 3、打印泛化误差可控部分最小值时对应的 n_estimators/subsample/learning_rate/gamma 参数取值； 2、并打印泛化误差可控部分最小值时对应的R2值； 3、并打印泛化误差可控部分最小值时对应的R2方差值； 4、并打印泛化误差可控部分最小值
    # print(axisx[ge_all_test.index(min(ge_all_test))],rs_all_test[ge_all_test.index(min(ge_all_test))], var_all_test[ge_all_test.index(min(ge_all_test))], min(ge_all_test))
    print("泛化误差可控部分最小值时对应的%s参数取值:%f； 泛化误差可控部分最小值时对应的R2值:%f； 泛化误差可控部分最小值时对应的R2方差值:%f； 泛化误差可控部分最小值:%f" % (
    axisx[ge_all_test.index(min(ge_all_test))], rs_all_test[ge_all_test.index(min(ge_all_test))],
    var_all_test[ge_all_test.index(min(ge_all_test))], min(ge_all_test)))

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


def mae_score(y_ture, y_pred):
    return mean_absolute_error(y_true=y_ture, y_pred=y_pred)


'''
gamma是如何控制过拟合？（gamma/γ复杂度惩罚项： 必调超参数）
1、gamma/γ复杂度惩罚项作用： 控制训练集上的训练：即，降低训练集上的表现（R^2降低、MSE升高），从而使训练集表现 和 测试集的表现 逐步趋近。
2、gamma不断增大，训练集R^2降低、MSE升高，训练集表现 和 测试集的表现 逐步趋近；但随着gamma不断增大，测试集也会出现R^2降低、MSE升高 的 欠拟合情况。所以，需要找到gamma的平衡点。
3、gamma主要是用来 降低模型复杂度、提高模型泛化能力的（防止过拟合）；不是用来提高模型准确性的（降低欠拟合）。
'''
# num_round/n_estimators 与 gamma/γ复杂度惩罚项 学习曲线： （xgboost原生交叉验证类： xgboost.cv）
# 多个gamma/γ复杂度惩罚项参数  分别对  训练集/测试集 X轴随着num_round/n_estimators（树的数量）增加，Y轴评估指标曲线趋势（类似于learning_curve基于样本量的学习曲线）
# 代码在： 1、10_2_XGBoost.py 中 “7.3、xgboost原生交叉验证类： xgboost.cv”； 2、10_3_XGBoost.py
# 建议优先使用这种学习曲线的调参方式（和 最终的调参方式 代码相同）
'''
gamma/γ复杂度惩罚项 学习曲线 使用：
1、将gamma/γ = 0： 使用之前经过“方差与泛化误差学习曲线”初步确定的超参数num_round/n_estimators（树的数量），看随着X轴（树数量的增加），评估指标曲线（默认RMSE）趋势。
如果 Y轴的评估指标曲线（默认RMSE）趋势 提前在 X轴达到num_round/n_estimators（树的数量）值之前就趋于平稳了，则可以从新选择 小于 num_round/n_estimators（树的数量）超参数的值（这一步从新选择num_round/n_estimators数量的操作是可选的）
2、gamma/γ = 20（例如），看 train和test的评估指标曲线之间的距离 是否较 gamma/γ = 0 时的距离缩短了（抑止过拟合）？ 正常情况下应该是缩短了，证明 gamma/γ复杂度惩罚项 生效了。
3、但需注意的是： gamma/γ = 20时的test的评估指标曲线 必须最少 与 gamma/γ = 0时的test的评估指标曲线 重合（非常接近），意思就是测试集效果不能降低（如果测试集效果能更好当然更好了）。

4、gamma/γ复杂度惩罚项作用： （其实就是上面“gamma是如何控制过拟合？”中所述）
4.1、控制训练集上的训练：即，降低训练集上的表现（R^2降低、MSE升高），从而使训练集表现 和 测试集的表现 逐步趋近。
4.2、降低模型复杂度、提高模型泛化能力的（防止过拟合）；不是用来提高模型准确性的（降低欠拟合）。
也就是之前说的： 基于超参数：比较现实的 目标 是将训练集效果降低，从而避免过拟合
4.3、但需注意的是： gamma不断增大，训练集R^2降低、MSE升高，训练集表现 和 测试集的表现 逐步趋近；但随着gamma不断增大，测试集也会出现R^2降低、MSE升高 的 欠拟合情况。所以，需要找到gamma的平衡点。
'''


def learning_curve_xgboost(X, y, param1, param2=None, num_round=300, metric="rmse", n_fold=5, axe=None,
                           set_ylim_top=None):
    import xgboost as xgb

    dfull = xgb.DMatrix(X, y)  # 为了便捷，使用全数据

    # X轴一定是num_round：树的数量。 Y轴：回归默认均方误差；分类默认error
    time0 = time()
    cvresult1 = xgb.cv(param1, dfull, num_boost_round=num_round, metrics=(metric), nfold=n_fold)
    print(time() - time0)

    if param2 is not None:
        time0 = time()
        cvresult2 = xgb.cv(param2, dfull, num_boost_round=num_round, metrics=(metric), nfold=n_fold)
        print(time() - time0)

    #    print(cvresult1)
    '''
    cvresult1[0]： 测试集rmse均值
    cvresult1[1]： 测试集rmse标准差
    cvresult1[2]： 训练集rmse均值
    cvresult1[3]： 训练集rmse标准差
    '''

    end_temp = num_round + 1  # X轴显示： num_round/n_estimators（树的数量）

    if axe is None:
        fig, axe = plt.subplots(1, figsize=(15, 8))
    axe.grid()

    temp_label1 = ""
    if "max_depth" in param1.keys() and param1['max_depth'] > 0:
        temp_label1 = ", max_depth=" + str(param1['max_depth'])
    elif "gamma" in param1.keys() and param1['gamma'] >= 0:
        temp_label1 = ", gamma=" + str(param1['gamma'])

    axe.plot(range(1, end_temp), cvresult1.iloc[:, 2], c="red", label="train" + temp_label1)
    axe.plot(range(1, end_temp), cvresult1.iloc[:, 0], c="orange", label="test" + temp_label1)

    if param2 is not None:
        temp_label2 = ""
        if "max_depth" in param1.keys() and param1['max_depth'] > 0:
            temp_label2 = ", max_depth=" + str(param1['max_depth'])
        elif "gamma" in param1.keys() and param1['gamma'] >= 0:
            temp_label2 = ", gamma=" + str(param1['gamma'])

        axe.plot(range(1, end_temp), cvresult2.iloc[:, 2], c="green", label="train" + temp_label2)
        axe.plot(range(1, end_temp), cvresult2.iloc[:, 0], c="blue", label="test" + temp_label2)

    if set_ylim_top is not None:
        axe.set_ylim(top=set_ylim_top)  # 截取Y轴最大值 进行显示

    axe.legend(fontsize="xx-large")

    plt.show()


# XGBoost调参方式： 非常重要（需必会） 建议优先使用这种学习曲线的调参方式： 3组参数/3个模型 在一个图中显示 进行评估指标对比调参
# 代码在 10_3_XGBoost.py 中 “二、学习曲线调参： （重点：调参方式）”


# -----------------------------2、基于超参数 结束-------------------------------

# In[]:
# ====================================学习曲线 结束==================================


# In[]:
# ====================================交叉验证 开始==================================
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


# 动态设置评估指标： （主要为了展示make_scorer用法： 装饰器函数）
def make_scorer_metrics_cv(model, train_X, train_y, is_log_transfer=False, cv=None, cv_type=1):
    from sklearn.metrics import mean_absolute_error, make_scorer

    if cv is None:
        if cv_type == 1:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 交叉验证模式
        elif cv_type == 2:
            cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)
        else:
            raise Exception('CV Type is Error')

    # 装饰器函数： 将 真实y值 和 预测y值 都取log
    def log_transfer(func):
        def wrapper(y, yhat):
            result = func(np.log(y), np.nan_to_num(np.log(yhat)))
            return result

        return wrapper

    # 它的意思是不是 非log的y 进来后，cv交叉验证的训练时就将y取log？ 否则只在评估时将y取log怕是没意义。

    # 使用make_scorer关键字，则评分函数需为动态评分函数（也就是直接导入的，不能是字符串名称形式）
    if is_log_transfer:
        temp_fun = log_transfer(mean_absolute_error)
    else:
        temp_fun = mean_absolute_error

    return CVS(model, train_X, train_y, verbose=1, cv=cv, scoring=make_scorer(temp_fun))


# 入参 X、y 都是矩阵格式
def stacking_cv_customize(stacking_model, X, y, ss):
    rs_train = []
    mse_train = []
    rs_test = []
    mse_test = []

    for train_index, test_index in ss.split(X, y):
        #        print("Train Index:", train_index, ",Test Index:", test_index)
        X_train, X_test = X[train_index], X[test_index]  # 矩阵的行索引： 取矩阵的一整行
        y_train, y_test = y[train_index], y[test_index]

        stacking_model.fit(X_train, y_train)
        y_predict_train = stacking_model.predict(X_train)
        rs_train.append(r2_score(y_train, y_predict_train))
        mse_train.append(MSE(y_train, y_predict_train))

        y_predict_test = stacking_model.predict(X_test)
        rs_test.append(r2_score(y_test, y_predict_test))
        mse_test.append(MSE(y_test, y_predict_test))

    rs_mean_train = np.mean(rs_train)
    rs_var_train = np.var(rs_train)
    ge_rs_train = (1 - rs_mean_train) ** 2 + rs_var_train  # 原来只有这个ge
    mse_mean_train = np.mean(mse_train)
    mse_var_train = np.var(mse_train)
    ge_mse_train = (1 - mse_mean_train) ** 2 + mse_var_train

    rs_mean_test = np.mean(rs_test)
    rs_var_test = np.var(rs_test)
    ge_rs_test = (1 - rs_mean_test) ** 2 + rs_var_test  # 原来只有这个ge
    mse_mean_test = np.mean(mse_test)
    mse_var_test = np.var(mse_test)
    ge_mse_test = (1 - mse_mean_test) ** 2 + mse_var_test

    print(rs_mean_train, rs_var_train, ge_rs_train, mse_mean_train, mse_var_train, ge_mse_train)
    print(rs_mean_test, rs_var_test, ge_rs_test, mse_mean_test, mse_var_test, ge_mse_test)

# In[]:
# ====================================交叉验证 结束==================================


