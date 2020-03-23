# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 09:53:24 2020

@author: dell
"""

import numpy as np
import pandas as pd
import os
import timeit

import FeatureTools as ft
import Tools_customize as tc

# In[]:
'''
代码在： Bat_lfm_learn.py
正负样本采样（原生读取.txt文件）
1、大于5分的为正样本，其余为负样本。以正样本和负样本中数量最少的一方为基准（一般情况下，负样本数量远大于正样本数量）
2、负样本以item的平均评分倒排序。
3、以map为临时存储结构，速度较快。
'''


def native_get_train_data_pos_neg(input_file, score_dict, split_char="::", title_num=None, score_thr=4.0,
                                  encoding="UTF-8"):
    if not os.path.exists(input_file):
        return {}
    neg_dict = {}
    pos_dict = {}
    train_data = []
    line_num = 0
    fp = open(input_file, encoding=encoding)
    for line in fp:
        if (title_num is not None) and (line_num <= title_num):
            line_num += 1
            continue
        item = line.strip().split(split_char)
        if len(item) < 4:
            continue
        userId, itemId, rating = item[0], item[1], item[2]
        if userId not in pos_dict:
            pos_dict[userId] = []
        if userId not in neg_dict:
            neg_dict[userId] = []
        if float(rating) > score_thr:
            pos_dict[userId].append((itemId, 1))
        else:
            score = score_dict.get(itemId, 0)
            neg_dict[userId].append((itemId, score))  # 设置item的平均评分，后续做排序
    fp.close

    print(pos_dict["1"])
    print(neg_dict["1"])
    for userId in pos_dict:
        data_num = min(len(pos_dict[userId]), len(neg_dict.get(userId, [])))
        if data_num > 0:
            train_data += [(userId, zuhe[0], zuhe[1]) for zuhe in pos_dict[userId]][:data_num]
        else:
            continue
        sorted_neg_list = tc.list_tuple_sorted(neg_dict[userId])[:data_num]
        train_data += [(userId, zuhe[0], 0) for zuhe in sorted_neg_list]
        if userId == "1":
            print(len(pos_dict[userId]))
            print(len(neg_dict[userId]))
            print(sorted_neg_list)
            print(len(sorted_neg_list))

    return train_data


# In[]:
def get_train_data_pos_neg(rating_df, score_df):
    rating_df_tmp = rating_df.merge(score_df, left_on='item_id', right_on='item_id', how='left')
    rating_df_tmp["label"] = rating_df_tmp["rating"].map(lambda x: 1 if x > 4.0 else 0)
    rating_df_tmp_pos = rating_df_tmp[rating_df_tmp["label"] == 1]
    rating_df_tmp_neg = rating_df_tmp[rating_df_tmp["label"] == 0]
    rating_df_tmp_neg_sort = tc.groupby_apply_sort(rating_df_tmp_neg, ["user_id"], ["rating_mean"], [False], False)
    train_data_df = pd.DataFrame(columns=rating_df_tmp.columns)

    tmp = rating_df_tmp.groupby(["user_id", "label"], as_index=False)["rating"].count()
    tmp1 = tmp.groupby(["user_id"], as_index=False)["rating"].count()
    user_id_tmp = tmp1[tmp1["rating"] == 2]
    tmp2 = tmp.merge(user_id_tmp, left_on='user_id', right_on='user_id')
    tmp3 = tmp2.groupby(["user_id"], as_index=False)["rating_x"].min()
    # 这里太耗时，只能寻找更快的方式
    time_start = timeit.default_timer()
    for item in tmp3.itertuples():
        # 必须使用 item[1] 或 item.user_id 取值， 不能使用： item["user_id"]；  索引取值： item[0] 或 item.Index
        temp1 = rating_df_tmp_pos[rating_df_tmp_pos["user_id"] == item.user_id][:item.rating_x]
        temp2 = rating_df_tmp_neg_sort[rating_df_tmp_neg_sort["user_id"] == item.user_id][:item.rating_x]
        train_data_df = pd.concat([train_data_df, temp1, temp2])
    time_end = timeit.default_timer()
    print(time_end - time_start)  # 324.5745487

    # 检查抽样情况：
    tmp4 = train_data_df.groupby(["user_id", "label"], as_index=False)["rating"].count()
    agg = {'rating_count1': lambda x: len(set(x)), 'rating_count2': lambda x: x.nunique()}
    tmp5 = tc.groupby_agg_oneCol(tmp4, ["user_id"], "rating", agg, as_index=False)
    print(tmp5[(tmp5["rating_count1"] > 1) | (tmp5["rating_count2"] > 1)])

    return train_data_df


# In[]:
# 由于原始样本量太大，无法使用基于P值的构建模型的方案，因此按照区进行分层抽样 （代码在： sndHsPr.py）
# dat0 = datall.sample(n=2000, random_state=1234).copy()
# dat01 = get_sample(dat0, sampling="stratified", k=400, stratified_col=['dist'])
def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    """
    对输入的 dataframe 进行抽样的函数

    参数:
        - df: 输入的数据框 pandas.dataframe 对象

        - sampling:抽样方法 str
            可选值有 ["simple_random", "stratified", "systematic"]
            按顺序分别为: 简单随机抽样、分层抽样、系统抽样

        - k: 抽样个数或抽样比例 int or float
            (int, 则必须大于0; float, 则必须在区间(0,1)中)
            如果 0 < k < 1 , 则 k 表示抽样对于总体的比例
            如果 k >= 1 , 则 k 表示抽样的个数；当为分层抽样时，代表每层的样本量

        - stratified_col: 需要分层的列名的列表 list
            只有在分层抽样时才生效

    返回值:
        pandas.dataframe 对象, 抽样结果
    """
    import random
    import pandas as pd
    from functools import reduce
    import numpy as np
    import math

    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n = True
        if sampling is "stratified":
            alln = k * df.groupby(by=stratified_col)[stratified_col[0]].count().count()  # 有问题的
            # alln=k*df[stratified_col].value_counts().count()
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n = False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)

    # print(k)

    if sampling is "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        res_df = df.iloc[idx, :].copy()
        return res_df

    elif sampling is "systematic":
        print("使用系统抽样")
        step = len_df // k + 1  # step=len_df//k-1
        start = 0  # start=0
        idx = range(len_df)[start::step]  # idx=range(len_df+1)[start::step]
        res_df = df.iloc[idx, :].copy()
        # print("k=%d,step=%d,idx=%d"%(k,step,len(idx)))
        return res_df

    elif sampling is "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.in1d(stratified_col, df.columns)), "请检查输入的列名"

        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n == True:
            group_k = grouped.map(lambda x: k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))

        res_df = df.head(0)
        for df_idx in group_k.index:
            df1 = df
            if len(stratified_col) == 1:
                df1 = df1[df1[stratified_col[0]] == df_idx]
            else:
                for i in range(len(df_idx)):
                    df1 = df1[df1[stratified_col[i]] == df_idx[i]]
            idx = random.sample(range(len(df1)), group_k[df_idx])
            group_df = df1.iloc[idx, :].copy()
            res_df = res_df.append(group_df)
        return res_df

    else:
        raise AssertionError("sampling is illegal")


