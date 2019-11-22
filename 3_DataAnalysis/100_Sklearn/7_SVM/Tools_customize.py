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
    dataDf1=pd.DataFrame({'lkey':['foo','bar','baz','foo'],
                     'value':[1,2,3,4]})
    dataDf2=pd.DataFrame({'rkey':['foo','bar','qux','bar'],
                         'value':[5,6,7,8]})
    print(dataDf1)
    print(dataDf2)

    # 内连接： 两DataFrame都必须有字段
    dataIn = dataDf1.merge(dataDf2, left_on='lkey',right_on='rkey') # 是按 字段 连接，不是按值。

    # 右连接： 以右边DataFrame字段为准
    dataR = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey',how='right')

    # 左连接： 以左边DataFrame字段为准
    dataL = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey',how='left')

    # 全链接： 两边都统计 
    dataQ = dataDf1.merge(dataDf2, left_on='lkey', right_on='rkey', how='outer')


# 简单的将2个DataFrame列向合并
def concat_test(df1, df2):
    return pd.concat([df1, df2], axis=1)


# In[]:
# 二、groupby
# 1、按count()统计，并将结果展开为DataFrame
def groupby_size(data, cols):
    if type(cols) == list:
        return data.groupby(cols).size().reset_index() # groupby_result.size() == groupby_result["X"].count()；但.count()的.reset_index()麻烦
    else:
        raise Exception('cols Type is Error')


# 2、将groupby的结果 转换为 dict
def groupby_to_dict(df):
    from collections import defaultdict
    
    user_count = df.groupby("user")["event"].count()
    
    user_count_index = user_count[user_count > 2].index.tolist()
    
    u = df["user"].isin(user_count_index)
    #df_train["user"].map(lambda x : x in user_count_index)
    
    w = df.loc[u, ["user","event"]].sort_values(by="user")
    
    z = w.groupby("user")
    
    eventsForUser = defaultdict(set)
    
    i = 0
    for key, val in z:
        print(key)
        print(val)
        val.apply(lambda x : eventsForUser[x[0]].add(x[1]), axis=1) # axis=1按列统计； 默认axis=0按行统计
        i = i + 1
        if i == 2:
            break





