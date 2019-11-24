# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:16:16 2019

@author: dell
"""

import numpy as np
import pandas as pd


# In[]:
# 一、表连接
# https://blog.csdn.net/gdkyxy2013/article/details/80785361
def merge_test():
    dataDf1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
                            'value': [1, 2, 3, 4]})
    dataDf2 = pd.DataFrame({'rkey': ['foo', 'bar', 'qux', 'bar'],
                            'value': [5, 6, 7, 8]})
    print(dataDf1)
    print(dataDf2)

    # 内连接： 两DataFrame都必须有字段
    dataIn = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey')

    # 右连接： 以右边DataFrame字段为准
    dataR = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='right')

    # 左连接： 以左边DataFrame字段为准
    dataL = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='left')

    # 全链接： 两边都统计
    dataQ = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='outer')

    # 全链接： 两边都统计
    # on：指的是用于连接的列索引名称，必须存在于左右两个DataFrame中，如果没有指定且其他参数也没有指定，则以两个DataFrame列名交集作为连接键；


#    dataQ_On = dataDf1.merge(dataDf2, on=["lkey","rkey"], how='outer')


# 简单的将2个DataFrame列向合并： pd.concat 在 def consolidated_data_col(train_X, train_y, axis=1): 函数中


# In[]:
# 二、groupby
# https://www.jianshu.com/p/a18fa2074ca4
# https://www.jianshu.com/p/42f1d2909bb6
# https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

# 1、传统groupby
'''
aggs = {'3_total_fee' : [np.min, np.max, np.mean, np.sum], '4_total_fee' : np.sum}
data_group = tc.groupby_agg(data[0:10], ["1_total_fee", "2_total_fee"], aggs)
'''


def groupby_agg(data, group_cols, aggs, as_index=True):  # group_keys在普通groupby中不生效
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    elif type(aggs) != dict:
        raise Exception('aggs Type is Error, must dict')

    # 例子： aggs = {'3_total_fee' : [np.min, np.max, np.mean, np.sum], '4_total_fee' : np.sum}
    # aggs中必须是 DataFrame中存在的、 待统计的特征。
    data_group = data.groupby(group_cols, as_index=as_index).agg(aggs)
    data_group.columns = ['_'.join(col).strip() for col in data_group.columns.values]
    return data_group


# 分组后 按指定特征进行排序
# groupby 和 apply 配合使用，只有group_keys关键字生效，as_index不适用。
# result = tc.groupby_apply_sort(result, ["Feature_1"], ["Correlation_Coefficient"], [False], False)
'''
使用分组排序：外层排序特征Feature_1_sort没用。 因为apply函数是按每个分组标签划分之后，再按该组内的特征进行排序，控制不了分组标签排序。
且 每个分组标签 对应的 外层排序特征Feature_1_sort 都相同，没有意义。 但奇怪的是单独使用Feature_1_sort排序时，会带动其他数值类型特征进行排序...
'''


def groupby_apply_sort(data, group_cols, sort_cols, ascendings, group_keys=False):  # group_keys默认True
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    elif type(sort_cols) != list:
        raise Exception('sort_cols Type is Error, must list')
    elif type(ascendings) != list:
        raise Exception('ascendings Type is Error, must list')

    return data.groupby(group_cols, group_keys=group_keys).apply(
        lambda x: x.sort_values(by=sort_cols, ascending=ascendings))


# 1、按count()统计，并将结果展开为DataFrame
def groupby_size(data, cols):
    if type(cols) == list:
        return data.groupby(
            cols).size().reset_index()  # groupby_result.size() == groupby_result["X"].count()；但.count()的.reset_index()麻烦
    else:
        raise Exception('cols Type is Error')


# 2、将groupby的结果 转换为 dict
def groupby_to_dict(df):
    from collections import defaultdict

    user_count = df.groupby("user")["event"].count()

    user_count_index = user_count[user_count > 2].index.tolist()

    u = df["user"].isin(user_count_index)
    # df_train["user"].map(lambda x : x in user_count_index)

    w = df.loc[u, ["user", "event"]].sort_values(by="user")

    z = w.groupby("user")

    eventsForUser = defaultdict(set)

    i = 0
    for key, val in z:
        print(key)
        print(val)  # 分组标签（元组） → DataFrame的原索引 → DataFrame的原值
        val.apply(lambda x: eventsForUser[x[0]].add(x[1]), axis=1)  # axis=1按列统计； 默认axis=0按行统计
        i = i + 1
        if i == 2:
            break


# 特殊例子
'''
def special_groupby_example(data):
    kkk = data[0:10].groupby(["1_total_fee", "2_total_fee"])[["3_total_fee","4_total_fee"]]

    # 冒号前sum是现在新加字段名， np.sum求和 分别作用于 "3_total_fee","4_total_fee"。
    # sum新加字段名 在 原始字段名之上 形成： ('sum', '3_total_fee')、 ('sum', '4_total_fee')
    #kkk = kkk.agg({'sum':np.sum})

    # 冒号前sum是现在新加字段名， np.sum求和、np.mean均值 都分别作用于 "3_total_fee","4_total_fee"。
    # sum新加字段名 在 原始字段名之上 形成： ('sum', '3_total_fee')、 ('sum', '4_total_fee')、('mean', '3_total_fee')、 ('mean', '4_total_fee')
    #kkk = kkk.agg({'sum':np.sum,"mean":np.mean})

    # 冒号前是原始字段名， 则字段各自执行自身的聚合操作
    kkk = kkk.agg({'3_total_fee':np.sum,"4_total_fee":np.mean})


    kkk.columns = ['_'.join(col).strip() for col in kkk.columns.values]
    print(kkk)
'''

# statistical_col待统计特征为单个特征，相当于为单个统计特征 新增统计字段。
'''
agg = {'sum':np.sum,"mean":np.mean}
data_group = tc.groupby_agg_oneCol(data, ["1_total_fee", "2_total_fee"], "3_total_fee", agg)
'''


def groupby_agg_oneCol(data, group_cols, statistical_col, agg):
    if type(group_cols) != list:
        raise Exception('group_col Type is Error, must list')
    if type(statistical_col) != str:
        raise Exception('statistical_col Type is Error, must str')
    if type(agg) != dict:
        raise Exception('aggs Type is Error, must dict')

    data_group = data.groupby(group_cols)[statistical_col].agg(agg)
    return data_group



