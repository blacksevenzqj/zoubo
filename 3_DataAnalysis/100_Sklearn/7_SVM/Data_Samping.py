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




