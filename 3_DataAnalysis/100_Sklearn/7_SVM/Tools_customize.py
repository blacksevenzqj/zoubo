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

    # 内连接： 两DataFrame都必须有字段中的该类别
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

# 以下两行代码效果相同
#    pd.merge(pd.DataFrame(age_cut_grouped_good), pd.DataFrame(age_cut_grouped_bad), left_index=True, right_index=True)
#    pd.concat([age_cut_grouped_good, age_cut_grouped_bad], axis=1)


# 集合去重合并为一维列表：
# 列表排序： https://www.cnblogs.com/huchong/p/8296025.html
def set_union(seriers1, seriers2, reverse=False):
    if type(seriers1) is not pd.Series or type(seriers2) is not pd.Series:
        raise Exception('seriers1/seriers2 Type is Error, must Series')
    if (len(seriers1) != len(seriers2)):
        raise Exception('seriers1 len != seriers2 len')

    #    return sorted(set(pd.concat([seriers1, seriers2], axis=0)), reverse=reverse)
    return sorted(set(seriers1).union(seriers2), reverse=reverse)


# In[]:
# 二、groupby
# https://www.jianshu.com/p/a18fa2074ca4
# https://www.jianshu.com/p/42f1d2909bb6
# https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

# 1、传统groupby（不会统计 np.nan）
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


# statistical_col统计特征为单个特征（速度近10倍于groupby_apply）
'''
agg = {'bankcard_count':lambda x:len(set(x)), 'bank_phone_num':lambda x:x.nunique()}
agg = {'sum':np.sum,"mean":np.mean,"len":len}
data_group = tc.groupby_agg_oneCol(data, ["1_total_fee", "2_total_fee"], "3_total_fee", agg)
'''


def groupby_agg_oneCol(data, group_cols, statistical_col, agg, as_index=True):
    if type(group_cols) != list:
        raise Exception('group_col Type is Error, must list')
    if type(statistical_col) != str:
        raise Exception('statistical_col Type is Error, must str')
    if type(agg) != dict:
        raise Exception('aggs Type is Error, must dict')

    data_group = data.groupby(group_cols, as_index=as_index)[statistical_col].agg(agg)
    return data_group


# 按count()统计，并将结果展开为DataFrame
def groupby_size(data, cols):
    if type(cols) == list:
        return data.groupby(
            cols).size().reset_index()  # groupby_result.size() == groupby_result["X"].count()；但.count()的.reset_index()麻烦
    else:
        raise Exception('cols Type is Error')


# 将groupby的结果 转换为 dict
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


# tc.groupby_apply_count(train_order, ["id"], 'type_pay', '在线支付', 'type_pay_zaixian')
def groupby_apply_conditionCount(data, group_cols, statistics_col, condition, reset_index_name, group_keys=False,
                                 inplace=False):
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')

    # .reset_index(name=reset_index_name) 其中的 name=reset_index_name 是重命名 apply新生成的统计列的名称
    return data.groupby(group_cols, group_keys=group_keys).apply(
        lambda x: x[x[statistics_col] == condition][statistics_col].count()).reset_index(inplace=inplace,
                                                                                         name=reset_index_name)


# tc.groupby_apply_nunique(train_recieve, ["id"], ["fix_phone"])
def groupby_apply_nunique(data, group_cols, statistics_cols, group_keys=False):
    if type(group_cols) != list:
        raise Exception('group_cols Type is Error, must list')
    elif type(statistics_cols) != list:
        raise Exception('statistics_cols Type is Error, must list')

    return data.groupby(group_cols, group_keys=group_keys).apply(lambda x: x[statistics_cols].nunique())


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


# In[]:
# 三、透视表(pivotTab)
# https://blog.csdn.net/bqw18744018044/article/details/80015840
# index相当于分组key； margins总计
def pivot_table_statistical(df, statistical_cols, index=None, columns=None, aggfunc='mean', margins=True):
    #    pd.pivot_table(df, index=['产地','类别'], values=['价格', '数量'], aggfunc=np.mean) # values 相当于 statistical_cols 待统计字段
    return df.pivot_table(statistical_cols, index=index, columns=columns, aggfunc=aggfunc, margins=margins)


# 交叉表(crossTab)： 相当于 df.groupby([col1, col2])[X].count() 展开显示
def crossTab_statistical(df, col1, col2, margins=True):
    return pd.crosstab(df[col1], df[col2], margins=margins)




