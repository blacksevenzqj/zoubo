# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:38:49 2020

@author: dell
"""

import pandas as pd
import datetime
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
# import xgboost as xgb
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling
color = sns.color_palette()
sns.set_style('darkgrid')

from math import isnan
import FeatureTools as ft
ft.set_file_path(r"D:\视频教程\8、项目\项目列表\比赛\和鲸\携程酒店浏览客户流失概率预测")
import Tools_customize as tc
import Binning_tools as bt

# In[]:
train_data = ft.readFile_inputData('userlostprob.txt', parse_dates = ['d','arrival'], sep='\t')
# In[]:
train_data.describe()
# In[]:
# 缺失值概览：
mis_val_table_ren_columns = ft.missing_values_table(train_data)
# In[]:
# 删除缺失值比列88%的列historyvisit_7ordernum
_, train_data = ft.missing_values_table(train_data, percent=88)
# In[]:
# 提前訂酒店天数，似乎日常上越晚訂越不會流失
train_data['advance_booking'] = (train_data['arrival'] - train_data['d']).dt.days
# In[]:
# 访问小时特征进行分桶： 分8个桶
time_period = [(0,(0,5)), (1,(5,6)), (2,(6,8)), (3,(8,11)), (4,(11,13)), (5,(13,17)), (6,(17,18)), (7,(18,24))]
# In[]：
def time_segmentation(x):
    midle_index = len(time_period) // 2
    start = time_period[midle_index][1][0]
    end = time_period[midle_index][1][1]
    if x >= start and x < end:
        return time_period[midle_index][0]
    elif x >= end:
        for i in range(midle_index+1, len(time_period)):
            start = time_period[i][1][0]
            end = time_period[i][1][1]
            if x >= start and x < end:
                return time_period[i][0]
    elif x < start:
        for i in range(midle_index-1, -1, -1):
            start = time_period[i][1][0]
            end = time_period[i][1][1]
            if x >= start and x < end:
                return time_period[i][0]
        
# In[]:
train_data['h_bins'] = train_data['h'].map(time_segmentation)
# In[]:
ft.writeFile_outData(train_data, r"D:\视频教程\8、项目\项目列表\比赛\和鲸\携程酒店浏览客户流失概率预测\train_data.csv")
# In[]:
new_data = train_data.drop(['d', 'arrival', 'h', 'sampleid'], axis=1)


# In[]:
# 1、异常值处理：
temp_describe = new_data.describe()
# In[]:
temp_describe = temp_describe.T
# In[]:
# 小于0的特征：
less_zero_cols = temp_describe[temp_describe['min'] < 0].index.tolist()
'''
共6列有负值，
但其中deltaprice_pre2_t1（24小时内已访问酒店价格与对手价差均值）正常，
所以其余5列，分别为
客户价值 （ctrip_profits）
客户价值近1年 (customer_value_profit）
用户偏好价格-24小时浏览最多酒店价格（delta_price1）、
用户偏好价格-24小时浏览酒店平均价格（delta_price2）、
当年酒店可订最低价（lowestprice）
'''
# In[]:
# 封装的：
less_zero_cols = ft.exception_values(new_data)
# In[]:
'''
客户价值近1年 (customer_value_profit）、客户价值 （ctrip_profits) 替换为0
用户偏好价格-24小时浏览最多酒店价格（delta_price1)、
用户偏好价格-24小时浏览酒店平均价格（delta_price2)、
当年酒店可订最低价（lowestprice）按中位数处理
'''
new_data.loc[new_data.customer_value_profit<0,'customer_value_profit'] = 0
new_data.loc[new_data.ctrip_profits<0,'ctrip_profits'] = 0
new_data.loc[new_data.delta_price1<0,'delta_price1'] = new_data['delta_price1'].median()
new_data.loc[new_data.delta_price2<0,'delta_price2'] = new_data['delta_price2'].median()
new_data.loc[new_data.lowestprice<0,'lowestprice'] = new_data['lowestprice'].median()

# In[]:
ft.writeFile_outData(new_data, "new_data.csv")
# In[]:
new_data = ft.readFile_inputData('new_data.csv')

# In[]:
# 2、缺失值处理：
mis_val_table_ren_columns = ft.missing_values_table(new_data)
mis_cols = mis_val_table_ren_columns.index.tolist()
'''
['historyvisit_visit_detailpagenum',
 'firstorder_bu',
 'decisionhabit_user',
 'historyvisit_totalordernum',
 'historyvisit_avghotelnum',
 'delta_price1',
 'delta_price2',
 'customer_value_profit',
 'ctrip_profits',
 'lasthtlordergap',
 'ordernum_oneyear',
 'ordercanceledprecent',
 'ordercanncelednum',
 'avgprice',
 'cr',
 'price_sensitive',
 'consuming_capacity',
 'starprefer',
 'businessrate_pre',
 'deltaprice_pre2_t1',
 'lastpvgap',
 'visitnum_oneyear',
 'commentnums_pre',
 'businessrate_pre2',
 'commentnums',
 'commentnums_pre2',
 'novoters_pre',
 'cityorders',
 'cancelrate_pre',
 'novoters_pre2',
 'lowestprice_pre',
 'uv_pre',
 'cr_pre',
 'lowestprice_pre2',
 'uv_pre2',
 'landhalfhours',
 'customereval_pre2',
 'novoters',
 'cancelrate',
 'cityuvs',
 'lowestprice',
 'hoteluv',
 'hotelcr']
'''
# In[]:
f, axes = plt.subplots(1,2, figsize=(23, 8))
ft.con_data_distribution(new_data, 'historyvisit_visit_detailpagenum', axes)

# In[]:
f, axes = plt.subplots(1,2, figsize=(23, 8))
ft.con_data_distribution(new_data, 'firstorder_bu', axes)

# In[]:
f, axes = plt.subplots(1,2, figsize=(23, 8))
ft.con_data_distribution(new_data, 'decisionhabit_user', axes)

# In[]:
f, axes = plt.subplots(1,2, figsize=(23, 8))
ft.con_data_distribution(new_data, 'historyvisit_totalordernum', axes)

# In[]:
# 趋于正态分布的字段，用均值填充：字段有businesstate_pre2,cancelrate_pre,businessrate_pre；



# In[]:
# 右偏分布的字段，用中位数填充





