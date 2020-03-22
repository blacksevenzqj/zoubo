# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:24:26 2019

@author: dell
"""

import pandas as pd
import datetime
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')

from math import isnan
import FeatureTools as ft

ft.set_file_path(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\2_DataFountain\2_Immediate-AI-Global-Challenger-Competition\data\AI_risk_train_V3.0_new")
import Tools_customize as tc
import Binning_tools as bt


# In[]:
def xgb_valid(train_set_x, train_set_y):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'silent': 1,
              'nthread': 8
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    model = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, metrics={'auc'}, seed=10)
    print(model)


def xgb_feature(train_set_x, train_set_y, test_set_x, test_set_y):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'silent': 1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    return predict, model


# In[]:
# 一、表读取
# In[]:
# 1、train_target表： id表示 唯一用户 申请贷款唯一编号； target表示 逾期标识。 后面构建的 训练集行维度 都以 train_target表 为基准。
train_target = pd.read_csv('train_target.csv', parse_dates=['appl_sbm_tm'])
# In[]:
print(train_target.loc[0, "appl_sbm_tm"], type(train_target.loc[0, "appl_sbm_tm"]))
# In[]:
# train_target 的 id是唯一的（id 代表唯一用户 的 唯一申请贷款编号）
agg = {'数量': len}
tc.groupby_agg_oneCol(train_target, ["id"], "target", agg, as_index=False).shape

# In[]
# 2、train_auth_info表（行维度等于train_target表，可直接连接）
train_auth = ft.readFile_inputData(train_name='train_auth_info.csv', parse_dates=['auth_time'])
# In[]:
ft.missing_values_table(train_auth)
# In[]:
# 120929是唯一用户数量； train_auth_info表示用户认证信息（同一个用户 一条认证信息）
agg = {'数量': len}
train_auth_id_only = tc.groupby_agg_oneCol(train_auth, ["id"], "id_card", agg, as_index=True)
# In[]:
# 导入字段类型检测
print(train_auth["id_card"].dtypes, train_auth.loc[0, "id_card"], type(train_auth.loc[0, "id_card"]),
      train_auth.loc[0, "id_card"] is np.nan)
print(train_auth["id_card"].dtypes, train_auth.loc[1, "id_card"], type(train_auth.loc[1, "id_card"]),
      train_auth.loc[1, "id_card"] is np.nan)
print(train_auth["id_card"].dtypes, train_auth.loc[3, "id_card"], type(train_auth.loc[3, "id_card"]),
      train_auth.loc[3, "id_card"] is np.nan)
print(train_auth["id_card"].dtypes, train_auth.loc[5, "id_card"], type(train_auth.loc[5, "id_card"]),
      train_auth.loc[5, "id_card"] is np.nan)
print(train_auth["id_card"].dtypes, train_auth.loc[7, "id_card"], type(train_auth.loc[7, "id_card"]),
      train_auth.loc[7, "id_card"] is np.nan)
print(train_auth["id_card"].dtypes, train_auth.loc[8, "id_card"], type(train_auth.loc[8, "id_card"]),
      train_auth.loc[8, "id_card"] is np.nan)
print(train_auth["id_card"].dtypes, train_auth.loc[14, "id_card"], type(train_auth.loc[14, "id_card"]),
      train_auth.loc[14, "id_card"] is np.nan)
# In[]:
print(train_auth["auth_time"].dtypes, train_auth.loc[0, "auth_time"], type(train_auth.loc[0, "auth_time"]),
      train_auth.loc[0, "auth_time"] is np.nan, train_auth.loc[0, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[1, "auth_time"], type(train_auth.loc[1, "auth_time"]),
      train_auth.loc[1, "auth_time"] is np.nan, train_auth.loc[1, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[5, "auth_time"], type(train_auth.loc[5, "auth_time"]),
      train_auth.loc[5, "auth_time"] is np.nan, train_auth.loc[5, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[7, "auth_time"], type(train_auth.loc[7, "auth_time"]),
      train_auth.loc[7, "auth_time"] is np.nan, train_auth.loc[7, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[8, "auth_time"], type(train_auth.loc[8, "auth_time"]),
      train_auth.loc[8, "auth_time"] is np.nan, train_auth.loc[8, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[15, "auth_time"], type(train_auth.loc[15, "auth_time"]),
      train_auth.loc[15, "auth_time"] is np.nan, train_auth.loc[15, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[17, "auth_time"], type(train_auth.loc[17, "auth_time"]),
      train_auth.loc[17, "auth_time"] is np.nan, train_auth.loc[17, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[28, "auth_time"], type(train_auth.loc[28, "auth_time"]),
      train_auth.loc[28, "auth_time"] is np.nan, train_auth.loc[28, "auth_time"] is pd.lib.NaT)
print(train_auth["auth_time"].dtypes, train_auth.loc[29, "auth_time"], type(train_auth.loc[29, "auth_time"]),
      train_auth.loc[29, "auth_time"] is np.nan, train_auth.loc[29, "auth_time"] is pd.lib.NaT)
# In[]:
# 2.1、为认证表建立临时表（目标1：求 缺失值 对 违约率 的影响分析； 目标2：创建 与业务相关新特征）
tmp_train_auth = pd.merge(train_target, train_auth, on=['id'], how="left")
# In[]:
# 2.1.1、缺失值 对 违约率 的影响分析
# 违约 和 未违约 对比
tmp_train_auth_null_target_merge = ft.missing_values_2categories_compare(tmp_train_auth, "target")
'''
从对比发现：
1、auth_time 的 target==1的缺失值百分比18.6 低于 target==0的缺失值百分比34.9（反向）
2、id_card 的 target==1的缺失值百分比15.2 低于 target==0的缺失值百分比31.9（反向）
'''
# In[]:
# 2.1.1.1、auth_time
auth_time_null, auth_time_null_stat, auth_time_null_stat2, auth_time_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_auth, "auth_time", "id", "target")
# In[]:
auth_time_null_stat  # auth_time 和 id_card 缺失值几乎相同
# In[]:
ft.df_change_colname(auth_time_null_stat2, {"f_null_count": "auth_time_null_count"})
# In[]
'''
注意，这里是反向的。auth_time缺失值时，违约比小，效果越好。
与 按target分的 auth_time趋势相同： target==1的缺失值百分比18.6 低于 target==0的缺失值百分比34.9
'''
auth_time_null_ratio  # 违约比：0.01534
# In[]:
# 新增 auth_time缺失值特征标识
# 因 行维度等于train_target表，也就是 每个id唯一用户 只有一条认证信息， 所以空值标识 才能直接取 0/1表示
tmp_train_auth['is_auth_time_authtable'] = ft.missValue_map_fillzo(tmp_train_auth, "auth_time", 2)

# In[]:
# 2.1.1.2、id_card
id_card_null, id_card_null_stat, id_card_null_stat2, id_card_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_auth, "id_card", "id", "target")
# In[]:
id_card_null_stat  # id_card 和 auth_time 缺失值几乎相同（auth_time缺失值标识可以代替id_card缺失值标识）
# In[]:
ft.df_change_colname(id_card_null_stat2, {"f_null_count": "id_card_null_count"})
# In[]
'''
注意，这里是反向的。id_card缺失值时，违约比小，效果越好。
与 按target分的 id_card趋势相同： target==1的缺失值百分比15.2 低于 target==0的缺失值百分比31.9
但 id_card 和 auth_time 缺失值几乎相同（auth_time缺失值标识可以代替id_card缺失值标识）
'''
id_card_null_ratio  # 违约比：0.01373

# In[]:
# 2.1.1.3、phone
phone_null, phone_null_stat, phone_null_stat2, phone_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_auth, "phone", "id", "target")
# In[]:
phone_null_stat  # auth_time缺失百分比98.7， id_card缺失百分比76.4
# In[]:
ft.df_change_colname(phone_null_stat2, {"f_null_count": "phone_null_count"})
# In[]
phone_null_ratio  # 违约比：0.04899

# In[]:
# 2.1.2、创建 与业务相关新特征（行维度 与 train_target相同）
tmp_train_auth['diff_day'] = tmp_train_auth.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days, axis=1)
# In[]:
# 是否认证时间在借贷时间前
tmp_train_auth['is_auth_time_before_borrow_time'] = tmp_train_auth.apply(
    lambda x: 0 if (x['is_auth_time_authtable'] == 0) else (1 if x['auth_time'] < x['appl_sbm_tm'] else 0), axis=1)
# In[]:
# 是否认证时间在借贷时间后
tmp_train_auth['is_auth_time_after_borrow_time'] = tmp_train_auth.apply(
    lambda x: 0 if (x['is_auth_time_authtable'] == 0) else (1 if x['auth_time'] > x['appl_sbm_tm'] else 0), axis=1)
# In[]:
# 认证时间在借贷时间前多少天
tmp_train_auth['auth_time_before_borrow_time_days'] = tmp_train_auth.apply(
    lambda x: 0 if (x['is_auth_time_before_borrow_time'] == 0) else (x['appl_sbm_tm'] - x['auth_time']).days, axis=1)
# In[]:
# 认证时间在借贷时间后多少天
tmp_train_auth['auth_time_after_borrow_time_days'] = tmp_train_auth.apply(
    lambda x: 0 if (x['is_auth_time_after_borrow_time'] == 0) else (x['auth_time'] - x['appl_sbm_tm']).days, axis=1)

# In[]:
# 2.2、train_target表 连接 train_auth表的 新特征
train_target_auth_NewF = tmp_train_auth.copy()
# In[]:
# 删除多余特征：
temp_columns = ["auth_time", "id_card", "phone"]  # 各特征已经生成新特征
train_target_auth_NewF.drop(columns=temp_columns, inplace=True)
# In[]:
# 最后统一填充缺失值： 需要先统计新增特征：列向（每一行）缺失值比例； 而后再填充缺失值。
# ft.missValue_all_fillna(train_target_auth_NewF, ["diff_day"], 0)


# In[]:
# 3、train_bankcard_info表（行维度大于train_target表，创建与train_target行维度相同的 新特征 进行连接）
train_bankcard = ft.readFile_inputData(train_name='train_bankcard_info.csv', encoding="GBK")
# In[]:
ft.missing_values_table(train_bankcard)  # bank_name 缺失206
# In[]:
# 120929是唯一用户数量； train_bankcard_info表示用户在各银行的开卡情况（同一个用户在1个银行开有多张卡）
agg = {'数量': len}
train_bankcard_id_only = tc.groupby_agg_oneCol(train_bankcard, ["id"], "bank_name", agg, as_index=True)
# In[]:
agg = {'数量': lambda x: x.count() > 10}  # 开卡量大于10张的用户
train_bankcard_id_lager_boolean = tc.groupby_agg_oneCol(train_bankcard, ["id"], "bank_name", agg, as_index=True)
# In[]:
# >10张卡：
# train_bankcard_id_lager_boolean[train_bankcard_id_lager_boolean["数量"] == True]
train_bankcard_id_only[train_bankcard_id_lager_boolean["数量"]]
# In[]:
train_bankcard_someOne = train_bankcard[train_bankcard["id"] == "20160329140001784605"]
# In[]:
# 导入字段类型检测
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[0, "tail_num"], type(train_bankcard.loc[0, "tail_num"]),
      train_bankcard.loc[0, "tail_num"] is np.nan, isnan(train_bankcard.loc[0, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[1, "tail_num"], type(train_bankcard.loc[1, "tail_num"]),
      train_bankcard.loc[1, "tail_num"] is np.nan, isnan(train_bankcard.loc[1, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[2, "tail_num"], type(train_bankcard.loc[2, "tail_num"]),
      train_bankcard.loc[2, "tail_num"] is np.nan, isnan(train_bankcard.loc[2, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[3, "tail_num"], type(train_bankcard.loc[3, "tail_num"]),
      train_bankcard.loc[3, "tail_num"] is np.nan, isnan(train_bankcard.loc[3, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[4, "tail_num"], type(train_bankcard.loc[4, "tail_num"]),
      train_bankcard.loc[4, "tail_num"] is np.nan, isnan(train_bankcard.loc[4, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[5, "tail_num"], type(train_bankcard.loc[5, "tail_num"]),
      train_bankcard.loc[5, "tail_num"] is np.nan, isnan(train_bankcard.loc[5, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[6, "tail_num"], type(train_bankcard.loc[6, "tail_num"]),
      train_bankcard.loc[6, "tail_num"] is np.nan, isnan(train_bankcard.loc[6, "tail_num"]))
print(train_bankcard["tail_num"].dtypes, train_bankcard.loc[7, "tail_num"], type(train_bankcard.loc[7, "tail_num"]),
      train_bankcard.loc[7, "tail_num"] is np.nan, isnan(train_bankcard.loc[7, "tail_num"]))
# In[]:
"增加特征"
# 3.1、唯一用户 有几家不同银行的卡
agg = {'bank_name_len': lambda x: len(set(x))}
bank_name_setlen = tc.groupby_agg_oneCol(train_bankcard, ["id"], "bank_name", agg, as_index=False)
# 唯一用户 有几张卡
agg = {'tail_num_len': len}
bank_num_len = tc.groupby_agg_oneCol(train_bankcard, ["id"], "tail_num", agg, as_index=False)

# In[]:
# 3.2、唯一用户 有几个不同电话号码
start_time = pd.Timestamp.now()
agg = {'bank_phone_num': lambda x: x.nunique()}
bank_phone_num_setlen = tc.groupby_agg_oneCol(train_bankcard, ["id"], "phone", agg,
                                              as_index=False)  # 速度近10倍于groupby_apply
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())
# start_time = pd.Timestamp.now()
# bank_phone_num_setlen = tc.groupby_apply_nunique(train_bankcard, ["id"], ["phone"])
# bank_phone_num_setlen.reset_index(inplace=True)
# diff = pd.Timestamp.now() - start_time
# print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 3.3、唯一用户 卡平均得分（卡类别得分取平均，怎么得到的 卡分数？）
train_bankcard['card_type_score'] = train_bankcard['card_type'].map(lambda x: 0.0154925 if x == '信用卡' else 0.02607069)
agg = {'card_type_score_mean': np.mean}
bank_card_type_score = tc.groupby_agg_oneCol(train_bankcard, ["id"], "card_type_score", agg, as_index=False)

# In[]:
# 3.4、分别有 几张信用卡、储蓄卡
agg = {'数量': len}
train_bankcard_card_type_num = tc.groupby_agg_oneCol(train_bankcard, ["id", "card_type"], "bank_name", agg,
                                                     as_index=False)
# In[]:
credit_card = train_bankcard_card_type_num[train_bankcard_card_type_num["card_type"] == "信用卡"]
credit_card.rename(columns={'数量': 'credit_card'}, inplace=True)
# In[]:
debit_card = train_bankcard_card_type_num[train_bankcard_card_type_num["card_type"] == "储蓄卡"]
debit_card.rename(columns={'数量': 'debit_card'}, inplace=True)

# In[]:
# 3.5、建立银行违约率临时表（目标1：求 缺失值 对 违约率 的影响； 目标2：求 银行违约率均值）
tmp_bank_target = pd.merge(train_target, train_bankcard, on=['id'], how='left')
# In[]:
# 3.5.1、单特征缺失值 对 违约率 的影响(可以 以 缺失值单特征作为查询主体)
print(ft.missing_values_table(tmp_bank_target))  # bank_name 缺失206
# In[]:
tmp_bank_target_bank_name_null = tmp_bank_target[tmp_bank_target["bank_name"].isnull()]
# In[]:
# id 和 target 是一一对应的（唯一用户 唯一违约状态）
# bank_name_null_count： 唯一用户 bank_name空值个数
# bank_name_null_target： 因为 targer是0/1值，所以按id分组后取均值 即可求得 每个id（唯一用户） 的 targer标签值（唯一违约状态）。
agg = {'bank_name_null_count': len, 'bank_name_null_target': np.mean}  # （速度近10倍于groupby_apply）
bank_name_null_stat = tc.groupby_agg_oneCol(tmp_bank_target_bank_name_null, ["id"], "target", agg, as_index=False)
# bank_name空值的203个的用户中，只有5个用户违约； 空值最多的有3个用户空2个bank_name值，但没有违约； 所以 不添加该特征。
# In[]:
# 3.5.2、a：银行违约率总计； b：一个用户 关联的 多家银行违约率均值
ccc = pd.crosstab(tmp_bank_target.bank_name, tmp_bank_target.target)
ccc['bank_default_rate'] = ccc[1] / (ccc[0] + 0.1)
ccc.reset_index(inplace=True)
# In[]:
tmp_bank_target = pd.merge(tmp_bank_target, ccc, on=['bank_name'], how='left')
# In[]:
start_time = pd.Timestamp.now()
agg = {'bank_default_rate_mean': np.mean}  # （速度近10倍于groupby_apply）
bank_name_default_rate_mean = tc.groupby_agg_oneCol(tmp_bank_target, ["id"], "bank_default_rate", agg, as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 3.6、train_target表 连接 train_bankcard表的 新特征（行维度 与 train_target相同）
# 有几家不同银行的卡
train_target_bankcard_NewF = pd.merge(train_target, bank_name_setlen, on=['id'], how='left')
# In[]:
# 有几张卡（将 总卡数 分为 信用卡 和 储蓄卡 两个特征）
# train_target_bankcard_NewF = pd.merge(train_target_bankcard_NewF, bank_num_len, on = ['id'], how='left')
# In[]:
# 信用卡
train_target_bankcard_NewF = pd.merge(train_target_bankcard_NewF, credit_card[["id", "credit_card"]], on=['id'],
                                      how='left')
# In[]:
# 储蓄卡
train_target_bankcard_NewF = pd.merge(train_target_bankcard_NewF, debit_card[["id", "debit_card"]], on=['id'],
                                      how='left')
# In[]:
# 有几个不同电话号码
train_target_bankcard_NewF = pd.merge(train_target_bankcard_NewF, bank_phone_num_setlen, on=['id'], how='left')
# In[]:
# 唯一用户 卡平均得分
train_target_bankcard_NewF = pd.merge(train_target_bankcard_NewF, bank_card_type_score, on=['id'], how='left')
# In[]:
# 唯一用户 银行违约率均值（一个用户 相关的 多家银行违约率均值）
train_target_bankcard_NewF = pd.merge(train_target_bankcard_NewF, bank_name_default_rate_mean, on=['id'], how='left')
# In[]:
# 最后统一填充缺失值： 需要先统计新增特征：列向（每一行）缺失值比例； 而后再填充缺失值。
# ft.missValue_all_fillna(train_target_bankcard_NewF, ["credit_card", "debit_card"], 0)


# In[]:
# 4、train_credit_info表（行维度等于train_target表，可直接连接）
train_credit = ft.readFile_inputData(train_name='train_credit_info.csv')
# In[]:
# 120929是唯一用户数量； train_credit_info表示网购平台信用信息（同一个用户 一条信用信息）
agg = {'数量': len}
train_credit_id_only = tc.groupby_agg_oneCol(train_credit, ["id"], "quota", agg, as_index=True)
# In[]:
# 导入字段类型检测
print(train_credit["quota"].dtypes, train_credit.loc[0, "quota"], type(train_credit.loc[0, "quota"]),
      train_credit.loc[0, "quota"] is np.nan, isnan(train_credit.loc[0, "quota"]))
print(train_credit["quota"].dtypes, train_credit.loc[1, "quota"], type(train_credit.loc[1, "quota"]),
      train_credit.loc[1, "quota"] is np.nan, isnan(train_credit.loc[1, "quota"]))
print(train_credit["quota"].dtypes, train_credit.loc[2, "quota"], type(train_credit.loc[2, "quota"]),
      train_credit.loc[2, "quota"] is np.nan, isnan(train_credit.loc[2, "quota"]))
print(train_credit["quota"].dtypes, train_credit.loc[43, "quota"], type(train_credit.loc[43, "quota"]),
      train_credit.loc[43, "quota"] is np.nan, isnan(train_credit.loc[43, "quota"]))
# In[]:
ft.missing_values_table(train_credit)

# In[]:
# 4.1、为网购平台信用表建立临时表（目标1：求 缺失值 对 违约率 的影响分析； 目标2：创建 与业务相关新特征）
tmp_train_credit = pd.merge(train_target, train_credit, on=['id'], how="left")
# In[]:
# 4.1.1、缺失值 对 违约率 的影响分析
# 缺失值标识（所有缺失值都相同，做一个缺失值特征标识）
tmp_train_credit['is_quota_credittable'] = ft.missValue_map_fillzo(tmp_train_credit, "quota", 3)
# In[]:
tmp_train_credit.loc[43, 'is_quota_credittable']  # 检测转换结果

# In[]:
# 4.1.2、创建 与业务相关新特征（行维度 与 train_target相同）
# 4.1.2.1、额度-使用值 = 剩余额度
tmp_train_credit['can_use_credittable'] = tmp_train_credit['quota'] - tmp_train_credit['overdraft']
# In[]:
# 4.1.2.2、评分的反序
# credit_score_max = np.max(tmp_train_credit['credit_score'])
# credit_score_min = np.min(tmp_train_credit['credit_score'])
# tmp_train_credit['credit_score_inverse'] = tmp_train_credit['credit_score'].map(lambda x :605-x)

# In[]:
# 4.1.2.3、评分的分箱
# 4.1.2.3.1.1、自动分箱可视化（画IV曲线：选择最优分箱个数）
afterbins, bins_woe, bins_iv, bins_pv, bins_woe_pv, bins_iv_pv, \
bins_spearmanr, bins_woe_spearmanr, bins_iv_spearmanr = \
    bt.graphforbestbin(tmp_train_credit, "credit_score", "target",
                       is_spearmanr=True)  # 注意： 返回的分箱区间在使用pd.cut函数之前，确保分箱区间bins的首尾分别为： -np.inf, np.inf
# In[]:
# 给出的卡方分箱区间 画WOE曲线
bin_list = bt.break_down_num_bins(bins_pv)
# kf_bins_woe == bins_woe_pv、 kf_bins_iv == bins_iv_pv（分箱区间相同，只是从算一遍而已）
kf_bins_woe, kf_bins_iv = bt.box_indicator_visualization(tmp_train_credit, "credit_score", "target", bin_list=bin_list)
print(bt.spearmanr_bins(tmp_train_credit, "credit_score", "target", bin_list))  # 斯皮尔曼分箱系数
# In[]:
# 给出的斯皮尔曼分箱区间 画WOE曲线
bin_list2 = bt.break_down_num_bins(bins_spearmanr)
# sp_bins_woe == bins_woe_spearmanr、 sp_bins_iv == bins_iv_spearmanr（分箱区间相同，只是从算一遍而已）
sp_bins_woe, sp_bins_iv = bt.box_indicator_visualization(tmp_train_credit, "credit_score", "target", bin_list=bin_list2)
print(bt.spearmanr_bins(tmp_train_credit, "credit_score", "target", bin_list2))  # 斯皮尔曼分箱系数

# In[]:
# 先看credit_score的直方图、盒须图（平台信用评分在200-400之间的违约率最高，并非信用评分越低就表征用户越可能逾期还款，所以需要做WOE分箱）
f, axes = plt.subplots(2, 2, figsize=(20, 18))
ft.class_data_distribution(tmp_train_credit, "credit_score", "target", axes)
# In[]:
# 4.1.2.3.1.2、盒须图分箱： 根据Y=1的盒须图区间 画WOE曲线
val_list = bt.box_whisker_diagram(tmp_train_credit, "credit_score", "target")
cs_bins_woe, cs_bins_iv = bt.box_indicator_visualization(tmp_train_credit, "credit_score", "target", bin_list=val_list)
print(bt.spearmanr_bins(tmp_train_credit, "credit_score", "target", val_list))  # 斯皮尔曼分箱系数

# In[]:
# 4.1.2.3.2、训练集 WOE数据 映射：
# 选择 斯皮尔曼分箱区间 因为从 WOE图 和 sp_bins_woe（bins_woe_spearmanr）统计表格中看出，各项指标都相对比较好。
bin_woe_map = sp_bins_woe["woe"]
ft.seriers_change_index(bin_woe_map, sp_bins_woe["bin_index"])
tmp_train_credit["credit_score_woe"] = bt.woe_mapping_simple(tmp_train_credit, "credit_score", "target", bin_list,
                                                             bin_woe_map)

# In[]:
# 4.2、train_target表 连接 train_credit_info表的 新特征
train_target_credit_NewF = tmp_train_credit.copy()
# In[]:
# 删除多余特征：
temp_columns = ["credit_score"]  # 各特征已经生成新特征
train_target_credit_NewF.drop(columns=temp_columns, inplace=True)

# In[]:
# 5、train_order_info表（行维度大于train_target表，创建与train_target行维度相同的 新特征 进行连接）
# '''
train_order = ft.readFile_inputData(train_name='train_order_info.csv', encoding="GBK")
# In[]:
print(train_order["amt_order"].dtypes, train_order.loc[0, 'amt_order'], type(train_order.loc[0, 'amt_order']),
      train_order.loc[0, 'amt_order'] is np.nan, isnan(train_order.loc[0, "amt_order"]))
print(train_order["amt_order"].dtypes, train_order.loc[8, 'amt_order'], type(train_order.loc[8, 'amt_order']),
      train_order.loc[8, 'amt_order'] is np.nan, isnan(train_order.loc[8, "amt_order"]))
print(train_order["unit_price"].dtypes, train_order.loc[0, 'unit_price'], type(train_order.loc[0, 'unit_price']),
      train_order.loc[0, 'unit_price'] is np.nan, isnan(train_order.loc[0, "unit_price"]))
print(train_order["unit_price"].dtypes, train_order.loc[17, 'unit_price'], type(train_order.loc[17, 'unit_price']),
      train_order.loc[17, 'unit_price'] is np.nan, isnan(train_order.loc[17, "unit_price"]))
# In[]:
print(train_order["time_order"].dtypes, train_order.loc[0, "time_order"], type(train_order.loc[0, "time_order"]),
      train_order.loc[0, "time_order"] is np.nan, train_order.loc[0, "time_order"] is pd.lib.NaT)
print(train_order["time_order"].dtypes, train_order.loc[8, "time_order"], type(train_order.loc[8, "time_order"]),
      train_order.loc[8, "time_order"] is np.nan, train_order.loc[8, "time_order"] is pd.lib.NaT)
# In[]:
# 两种数据类型： 1、字符串时间格式； 2、时间戳 所以才使用这种方式
train_order['time_order'] = ft.missValue_datatime(train_order, "time_order")
# In[]:
ft.missing_values_table(train_order)

# In[]:
# 5.1、为订单表建立临时表（目标1：求 缺失值 对 违约率 的影响分析； 目标2：创建 与业务相关新特征）
tmp_train_order = pd.merge(train_order, train_target, on=['id'])
# In[]:
# 5.1.1、缺失值 对 违约率 的影响分析
# 违约 和 未违约 对比
tmp_train_order_null_target_merge = ft.missing_values_2categories_compare(tmp_train_order, "target")
'''
从对比发现：
1、time_order、amt_order两特征后续要用，且 target==1 大于 target==0 的缺失值百分比。
2、name_rec_md5、type_pay 的 target==1的缺失值百分比 高于 target==0的缺失值百分比。
'''
# In[]:
# 5.1.1.1、time_order
time_order_null, time_order_null_stat, time_order_null_stat2, time_order_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_order, "time_order", "id", "target")
# In[]:
# 当time_order缺失值时，其他特征也几乎全部都是缺失值，除非违约比较大，否则独立出time_order缺失值标识 效果不明显。
time_order_null_stat
# In[]:
# 当time_order缺失值时，违约情况（有大量 缺失值累加和特征 的 用户并没有违约； 且 违约比很低）
ft.df_change_colname(time_order_null_stat2, {"f_null_count": "time_order_null_count"})
# In[]
time_order_null_ratio  # 违约比为：0.0287

# In[]:
# 5.1.1.2、amt_order
amt_order_null, amt_order_null_stat, amt_order_null_stat2, amt_order_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_order, "amt_order", "id", "target")
# In[]:
# 当amt_order缺失值时，比time_order缺失值多，也正是多出的这部分缺失值， 使amt_order违约比 比 time_order违约比 大
amt_order_null_stat
# In[]:
# 当 amt_order 缺失值时，违约情况（有大量 缺失值累加和特征 的 用户并没有违约； 且 违约比很低）
ft.df_change_colname(amt_order_null_stat2, {"f_null_count": "amt_order_null_count"})
# In[]
amt_order_null_ratio  # 违约比为：0.0310

# In[]:
# 5.1.1.3、name_rec_md5（以 name_rec_md5 为主要缺失值特征分析）
name_rec_md5_null, name_rec_md5_null_stat, name_rec_md5_null_stat2, name_rec_md5_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_order, "name_rec_md5", "id", "target")
# In[]:
# 当 name_rec_md5缺失值时，明显比 其他特征缺失值多，远高于time_order、amt_order缺失值，但 违约比 并未明显增大。
name_rec_md5_null_stat
# In[]:
# 当 name_rec_md5 缺失值时，违约情况（有大量 缺失值累加和特征 的 用户并没有违约； 且 违约比很低）
ft.df_change_colname(name_rec_md5_null_stat2, {"f_null_count": "name_rec_md5_null_count"})
# In[]
name_rec_md5_null_ratio  # 违约比为：0.0354

# In[]:
# 5.1.1.4、type_pay（以 type_pay 为主要缺失值特征分析）
type_pay_null, type_pay_null_stat, type_pay_null_stat2, type_pay_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_order, "type_pay", "id", "target")
# In[]:
# 当 type_pay缺失值时，明显比其他特征的缺失值多，远高于time_order、amt_order缺失值，但 违约比 并未明显增大。
type_pay_null_stat
# In[]:
# 当 type_pay 缺失值时，违约情况（有大量 缺失值累加和特征 的 用户并没有违约； 且 违约比很低）
ft.df_change_colname(type_pay_null_stat2, {"f_null_count": "type_pay_null_count"})
# In[]
type_pay_null_ratio  # 违约比为：0.03625

# In[]:
# 5.1.2、创建 与业务相关新特征（行维度 与 train_target相同）
# 借贷时间前 消费数据
tmp_train_order_before_appl_sbm_tm = tmp_train_order[tmp_train_order["time_order"] < tmp_train_order["appl_sbm_tm"]]
# In[]:
# 借贷时间后 消费数据
tmp_train_order_after_appl_sbm_tm = tmp_train_order[tmp_train_order["time_order"] > tmp_train_order["appl_sbm_tm"]]

# In[]:
# 借贷时间前有多少次购买
start_time = pd.Timestamp.now()
agg = {'before_appl_sbm_tm_howmany': len}  # （速度近10倍于groupby_apply）
before_appl_sbm_tm_howmany = tc.groupby_agg_oneCol(tmp_train_order_before_appl_sbm_tm, ["id"], "amt_order", agg,
                                                   as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())
# In[]:
# 借贷时间后有多少次购买
start_time = pd.Timestamp.now()
agg = {'after_appl_sbm_tm_howmany': len}  # （速度近10倍于groupby_apply）
after_appl_sbm_tm_howmany = tc.groupby_agg_oneCol(tmp_train_order_after_appl_sbm_tm, ["id"], "amt_order", agg,
                                                  as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 借贷时间前购买金额均值
start_time = pd.Timestamp.now()
agg = {'before_appl_sbm_tm_money_mean': np.mean}  # （速度近10倍于groupby_apply）
before_appl_sbm_tm_money_mean = tc.groupby_agg_oneCol(tmp_train_order_before_appl_sbm_tm, ["id"], "amt_order", agg,
                                                      as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())
# In[]:
# 借贷时间后购买金额均值
start_time = pd.Timestamp.now()
agg = {'after_appl_sbm_tm_money_mean': np.mean}  # （速度近10倍于groupby_apply）
after_appl_sbm_tm_money_mean = tc.groupby_agg_oneCol(tmp_train_order_after_appl_sbm_tm, ["id"], "amt_order", agg,
                                                     as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 借贷时间前购买金额最大值
start_time = pd.Timestamp.now()
agg = {'before_appl_sbm_tm_money_max': np.max}  # （速度近10倍于groupby_apply）
before_appl_sbm_tm_money_max = tc.groupby_agg_oneCol(tmp_train_order_before_appl_sbm_tm, ["id"], "amt_order", agg,
                                                     as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())
# In[]:
# 借贷时间前购买金额最小值
start_time = pd.Timestamp.now()
agg = {'before_appl_sbm_tm_money_min': np.min}  # （速度近10倍于groupby_apply）
before_appl_sbm_tm_money_min = tc.groupby_agg_oneCol(tmp_train_order_before_appl_sbm_tm, ["id"], "amt_order", agg,
                                                     as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 借贷时间后购买金额最大值
start_time = pd.Timestamp.now()
agg = {'after_appl_sbm_tm_money_max': np.max}  # （速度近10倍于groupby_apply）
after_appl_sbm_tm_money_max = tc.groupby_agg_oneCol(tmp_train_order_after_appl_sbm_tm, ["id"], "amt_order", agg,
                                                    as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())
# In[]:
# 借贷时间后购买金额最小值
start_time = pd.Timestamp.now()
agg = {'after_appl_sbm_tm_money_min': np.min}  # （速度近10倍于groupby_apply）
after_appl_sbm_tm_money_min = tc.groupby_agg_oneCol(tmp_train_order_after_appl_sbm_tm, ["id"], "amt_order", agg,
                                                    as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 没用的特征：（作为参考）
# 用户 最早/最晚 订单时间
# train_order_time_max = tmp_train_order.groupby(by=['id'], as_index=False)['time_order'].agg({'train_order_time_max':lambda x:max(x)})
# train_order_time_min = tmp_train_order.groupby(by=['id'], as_index=False)['time_order'].agg({'train_order_time_min':lambda x:min(x)})

# 用户 最早/最晚 订单时间 与 申请贷款时间 差值
# tmp_train_order['day_order_max'] = tmp_train_order.apply(lambda row: (row['appl_sbm_tm'] - row['train_order_time_max']).days,axis=1);
# tmp_train_order.drop(['train_order_time_max'], axis=1, inplace=True)
# tmp_train_order['day_order_min'] = tmp_train_order.apply(lambda row: (row['appl_sbm_tm'] - row['train_order_time_min']).days,axis=1);
# tmp_train_order.drop(['train_order_time_min'], axis=1, inplace=True)

# 用户 在线支付/货到付款 次数
# train_order_type_zaixian = tmp_train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='在线支付').values].count()).reset_index(name = 'type_pay_zaixian')
# train_order_type_huodao = tmp_train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='货到付款').values].count()).reset_index(name = 'type_pay_huodao')


# In[]:
# 5.2、train_target表 连接 train_order_info表的 新特征
# 5.2.1、新增缺失值标识
# 多特征缺失值： 新增 特征缺失值标识（time_order、amt_order、name_rec_md5、type_pay）
# 因 行维度大于train_target表，所以要使用groupby，也就是 每个id唯一用户 有多条订单信息， 所以空值标识 要使用累加和 表示
# id 和 target 是一一对应的（唯一用户 唯一违约状态）
# XXX_null_count： 唯一用户 bank_name空值个数
# XXX_null_target： 因为 targer是0/1值，所以按id分组后取均值 即可求得 每个id（唯一用户） 的 targer标签值（唯一违约状态）。不连接添加。
# In[]:
# time_order缺失值总计 新特征
# train_target_order_NewF = pd.merge(train_target, time_order_null_stat2[["id","time_order_null_count"]], on = ['id'], how='left')
# In[]:
# amt_order缺失值总计 新特征
# train_target_order_NewF = pd.merge(train_target_order_NewF, amt_order_null_stat2[["id","amt_order_null_count"]], on = ['id'], how='left')
# In[]:
# name_rec_md5缺失值总计 新特征
# train_target_order_NewF = pd.merge(train_target_order_NewF, name_rec_md5_null_stat2[["id","name_rec_md5_null_count"]], on = ['id'], how='left')
# In[]:
# type_pay缺失值总计 新特征
# train_target_order_NewF = pd.merge(train_target_order_NewF, type_pay_null_stat2[["id","type_pay_null_count"]], on = ['id'], how='left')
'''
各缺失值特征中 有大量 缺失值累加和特征 的 用户并没有违约，所以这些 缺失值标识特征 反倒会影响模型判断，最后不添加进训练集。
'''

# In[]:
# 5.2.2、与业务相关新特征
train_target_order_NewF = pd.merge(train_target, before_appl_sbm_tm_howmany, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, after_appl_sbm_tm_howmany, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, before_appl_sbm_tm_money_mean, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, after_appl_sbm_tm_money_mean, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, before_appl_sbm_tm_money_max, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, before_appl_sbm_tm_money_min, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, after_appl_sbm_tm_money_max, on=['id'], how='left')
train_target_order_NewF = pd.merge(train_target_order_NewF, after_appl_sbm_tm_money_min, on=['id'], how='left')

# In[]:
# 6、train_recieve_addr_info表（行维度大于train_target表，创建与train_target行维度相同的 新特征 进行连接）
train_recieve = pd.read_csv('train_recieve_addr_info.csv', encoding="GBK")
# In[]:
# 120929是唯一用户数量； train_recieve_addr_info表示用户收货情况（同一个用户，多条收货记录）
agg = {'数量': len}
train_recieve_id_only = tc.groupby_agg_oneCol(train_recieve, ["id"], "region", agg, as_index=True)
# In[]:
# 导入字段类型检测
print(train_recieve["region"].dtypes, train_recieve.loc[0, "region"], type(train_recieve.loc[0, "region"]),
      train_recieve.loc[0, "region"] is np.nan)
print(train_recieve["region"].dtypes, train_recieve.loc[1, "region"], type(train_recieve.loc[1, "region"]),
      train_recieve.loc[1, "region"] is np.nan)
print(train_recieve["region"].dtypes, train_recieve.loc[25327, "region"], type(train_recieve.loc[25327, "region"]),
      train_recieve.loc[25327, "region"] is np.nan)
# In[]:
print(train_recieve["addr_id"].dtypes, train_recieve.loc[0, "addr_id"], type(train_recieve.loc[0, "addr_id"]),
      train_recieve.loc[0, "addr_id"] is np.nan, isnan(train_recieve.loc[0, "addr_id"]))
print(train_recieve["addr_id"].dtypes, train_recieve.loc[1, "addr_id"], type(train_recieve.loc[1, "addr_id"]),
      train_recieve.loc[1, "addr_id"] is np.nan, isnan(train_recieve.loc[1, "addr_id"]))
print(train_recieve["addr_id"].dtypes, train_recieve.loc[25327, "addr_id"], type(train_recieve.loc[25327, "addr_id"]),
      train_recieve.loc[25327, "addr_id"] is np.nan, isnan(train_recieve.loc[25327, "addr_id"]))
# In[]:
ft.missing_values_table(train_recieve)
# In[]:
# 收货地址
train_recieve['first_name'] = train_recieve['region'].map(lambda x: x[:2] if x is not np.nan else x)
train_recieve['last_name'] = train_recieve['region'].map(lambda x: x[-1] if x is not np.nan else x)
# In[]:
# 交叉表观测： 一个用户 有几个收货地址（按 省份算： 有几条非空数据就有几个收货地址）
# 如果作为 新特征添加，列数太多，不考虑
tmp_tmp_recieve = tc.crossTab_statistical(train_recieve, 'id', 'first_name')
tmp_tmp_recieve = tmp_tmp_recieve.reset_index()
ft.df_change_colname(tmp_tmp_recieve, {"All": "recieve_addr_count"})
# In[]:
# 一个用户 有几个不同收货地址（按 省份算： 不统计np.nan； 0表示该条数据收货地址为np.nan，不进行统计）
agg = {'recieve_region_num': lambda x: x.nunique()}  # （速度近10倍于groupby_apply）
tmp_region_nunique = tc.groupby_agg_oneCol(train_recieve, ["id"], "first_name", agg, as_index=False)
# In[]:
# 一个用户 有几个不同的固定电话号码
# tmp_tmp_recieve_phone_count_unique = tc.groupby_apply_nunique(train_recieve, ["id"], ["fix_phone"])
# tmp_tmp_recieve_phone_count_unique = tmp_tmp_recieve_phone_count_unique.reset_index()
start_time = pd.Timestamp.now()
agg = {'recieve_fix_phone_num': lambda x: x.nunique()}  # （速度近10倍于groupby_apply）
tmp_tmp_recieve_phone_count_unique = tc.groupby_agg_oneCol(train_recieve, ["id"], "fix_phone", agg, as_index=False)
diff = pd.Timestamp.now() - start_time
print(diff, type(diff), diff.days, diff.seconds, diff.total_seconds())

# In[]:
# 6.1、为收货表建立临时表（目标1：求 缺失值 对 违约率 的影响分析； 目标2：创建 与业务相关新特征）
tmp_recieve_target = pd.merge(train_target, train_recieve, on=['id'], how="left")
# In[]:
# 6.1.1、缺失值 对 违约率 的影响分析
# 违约 和 未违约 对比
tmp_train_recieve_null_target_merge = ft.missing_values_2categories_compare(tmp_recieve_target, "target")
# In[]:
# 6.1.1.1、fix_phone
fix_phone_null, fix_phone_null_stat, fix_phone_null_stat2, fix_phone_null_ratio = \
    ft.feature_missing_value_analysis(tmp_recieve_target, "fix_phone", "id", "target")
# In[]:
# 当fix_phone缺失值时，明显比 其他特征缺失值多，远高于region缺失值，但 违约比 并未明显增大。
fix_phone_null_stat
# In[]:
# 当fix_phone缺失值时，违约情况（有大量 缺失值累加和特征 的 用户并没有违约； 且 违约比很低）
ft.df_change_colname(fix_phone_null_stat2, {"f_null_count": "fix_phone_null_count"})
# In[]
fix_phone_null_ratio  # 违约比：0.0264， 丢弃该缺失值特征标识

# In[]:
# 6.1.1.2、region
region_null, region_null_stat, region_null_stat2, region_null_ratio = \
    ft.feature_missing_value_analysis(tmp_recieve_target, "region", "id", "target")
# In[]:
# 当region缺失值时，其他特征也全部都是缺失值，独立出region缺失值标识 效果不明显。
region_null_stat
# In[]:
# 当region缺失值时，违约情况（有大量 缺失值累加和特征 的 用户并没有违约； 且 违约比很低）
ft.df_change_colname(region_null_stat2, {"f_null_count": "region_null_count"})
# In[]
region_null_ratio  # 违约比：0.0336， 丢弃该缺失值特征标识

# In[]:
# 6.1.2、创建 与业务相关新特征（行维度 与 train_target相同）
# a：收货省份违约率总计； b：一个用户 关联的 多个收货省份违约率均值
# 交叉表： 收货地址违约情况总计
ccc = pd.crosstab(tmp_recieve_target.first_name, tmp_recieve_target.target)
ccc['region_default_rate'] = ccc[1] / (ccc[0] + 0.1)
ccc.reset_index(inplace=True)
tmp_recieve_target = pd.merge(tmp_recieve_target, ccc, on=['first_name'], how='left')
# 一个用户 关联的 多个收货省份违约率均值
recieve_score_mean = tmp_recieve_target.groupby(by=['id'], as_index=False)['recieve_default_rate'].agg(
    {'recieve_default_rate_mean': np.mean})

# In[]:
# 6.2、train_target表 连接 train_recieve_addr_info表的 新特征
# 一个用户 有几个不同收货地址（按 省份算： 不统计np.nan； 0表示该条数据收货地址为np.nan，不进行统计）
# train_target_recieve_NewF = pd.merge(train_target, tmp_region_nunique, on = ['id'], how='left')
# In[]:
# 一个用户 有几个不同的固定电话号码
# train_target_recieve_NewF = pd.merge(train_target_recieve_NewF, tmp_tmp_recieve_phone_count_unique, on = ['id'], how='left')
# In[]:
# 一个用户 有几个不同的固定电话号码
# train_target_recieve_NewF = pd.merge(train_target_recieve_NewF, recieve_score_mean, on = ['id'], how='left')
'''
各新增特征中 recieve_region_num、recieve_fix_phone_num有大量 累加和特征 的 用户并没有违约，所以这些 新增特征 反倒会影响模型判断，最后不添加进训练集。
而 recieve_default_rate_mean 一个用户 关联的 多个收货省份违约率均值 也没有明显区别，也不添加进模型（除非再细化到县一级地址）。
'''

# In[]:
# 7、train_user_info（行维度等于train_target表，可直接连接）
train_user = pd.read_csv('train_user_info.csv', encoding="GBK")
# In[]:
# train_user 的 id是唯一的（id 代表唯一用户，相当于身份证）
agg = {'数量': len}
train_user_id_set = tc.groupby_agg_oneCol(train_user, ["id"], "sex", agg, as_index=True)
# In[]:
ft.missing_values_table(train_user)
# In[]:
train_user1 = train_user.copy()
# In[]:
error_list = ["--", "0-0-0", "1-1-1", "0000-00-00", "0001-1-1", "0001-01-01"]
# In[]:
for i in error_list:
    a = train_user1[train_user1['birthday'] == i]['birthday'].index.tolist()
    train_user1.loc[a, 'birthday'] = pd.lib.NaT

# In[]:
error_list1 = ["后", "null", "?", "？"]
# In[]:
# 找索引
for i in error_list1:
    a = train_user1[train_user1['birthday'].map(lambda x: i in str(x))]["birthday"].index.tolist()
    train_user1.loc[a, 'birthday'] = pd.lib.NaT
# In[]:
# 直接赋值
for i in error_list1:
    train_user1['birthday'] = train_user1['birthday'].map(lambda x: pd.lib.NaT if i in str(x) else x)

# In[]:
# re.match 返回 匹配对象 或 None
re_list = ["^(19|20)\d{2}-\d{1,2}-0", "^0-", "^-", "^(19|20)\d{2}-\d{1,2}-$"]
# In[]:
# 1、找索引
for i in re_list:
    a = train_user1[train_user1['birthday'].map(lambda x: re.match(i, str(x)) != None)]["birthday"].index.tolist()
    train_user1.loc[a, 'birthday'] = pd.lib.NaT
# In[]:
# 2、直接赋值
for i in re_list:
    train_user1['birthday'] = train_user1['birthday'].map(lambda x: pd.lib.NaT if (re.match(i, str(x))) else x)

# In[]:
# 查询出留下的正常数据
a = train_user1[train_user1['birthday'].map(lambda x: re.match("^(19|20)\d{2}-\d{1,2}-\d{1,2}", str(x)) != None)][
    "birthday"].index.tolist()
train_user1.loc[a, 'birthday']
# 其余不正常数据 赋值pd.lib.NaT
# 注意： 剩下的都是可以转换的正常数据。 本处程序会自动把把re.match匹配到的正常数据 也转换为 pd.Timestamp
train_user1['birthday'] = ft.missValue_datatime_match(train_user1, 'birthday')

# In[]:
# ft.writeFile_outData(train_user1, r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\2_DataFountain\2_Immediate-AI-Global-Challenger-Competition\MyLearn\birthday.csv", "GBK")

# In[]:
print(train_user1["birthday"].dtypes, train_user1.loc[0, "birthday"], type(train_user1.loc[0, "birthday"]))
print(train_user1["birthday"].dtypes, train_user1.loc[6, "birthday"], type(train_user1.loc[6, "birthday"]))
# In[]:
train_user = train_user1

# In[]:
# 7.1、为用户表建立临时表（目标1：求 缺失值 对 违约率 的影响分析； 目标2：创建 与业务相关新特征）
tmp_train_user = pd.merge(train_target, train_user, on=['id'], how="left")
# In[]:
# 7.1.1、缺失值 对 违约率 的影响分析
# 违约 和 未违约 对比
tmp_train_user_null_target_merge = ft.missing_values_2categories_compare(tmp_train_user, "target")
'''
从对比发现：
1、'industry','degree','merriage','id_card','income','hobby' 6个特征缺失值百分比90%以上，且target==1 和 target==0时变化不大，做 缺失值特征标识 没有多大效果。
2、birthday 的 target==1的缺失值百分比68.5 与 target==0的缺失值百分比67.3，相差无几，不做 缺失值特征标识
2、account_grade 的 target==1的缺失值百分比9.9 远低于 target==0的缺失值百分比45.2（反向）
'''
# In[]:
# 7.1.1.1、account_grade
account_grade_null, account_grade_null_stat, account_grade_null_stat2, account_grade_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_user, "account_grade", "id", "target")
# In[]:
account_grade_null_stat
# In[]:
ft.df_change_colname(account_grade_null_stat2, {"f_null_count": "account_grade_null_count"})
# In[]
'''
注意，这里是反向的。account_grade缺失值时，违约比小，效果越好。
与 按target分的 account_grade趋势相同： target==1的缺失值百分比9.9 远低于 target==0的缺失值百分比45.2
'''
account_grade_null_ratio  # 违约比：0.0063
# In[]:
# 新增 account_grade缺失值特征标识
tmp_train_user['account_grade'] = ft.missValue_map_conversion(tmp_train_user, 'account_grade', 2)  # 缺失值 统一转换
tmp_train_user['is_account_grade_usertable'] = ft.missValue_map_fillzo(tmp_train_user, "account_grade")

# In[]:
# 7.1.1.2、sex
sex_null, sex_null_stat, sex_null_stat2, sex_null_ratio = \
    ft.feature_missing_value_analysis(tmp_train_user, "sex", "id", "target")
# In[]:
# 当sex缺失值时，其他特征也几乎全部都是缺失值，除非违约比较大，否则独立出sex缺失值标识 效果不明显。
sex_null_stat
# In[]:
# 当sex缺失值时，违约情况（违约比很低）
ft.df_change_colname(sex_null_stat2, {"f_null_count": "sex_null_count"})
# In[]
sex_null_ratio  # 违约比：0.0449 丢弃该缺失值特征标识

# In[]:
# 7.1.2、创建 与业务相关新特征（行维度 与 train_target相同）
# 7.1.2.1、sex性别：
# 将性别的np.nan更换为“未知”（变相的 将 sex缺失值特征标识 的功能 实现）
ft.missValue_fillna(tmp_train_user, "sex", "未知")  # 将np.nan作为 分类特征 的一种类别，则无需进行 列向（每一行）缺失值比例 统计。
# In[]:
# 交叉表： sex性别 与 target 的关系
ccc = pd.crosstab(tmp_train_user["sex"], tmp_train_user["target"])
ccc['sex_default_rate'] = ccc[1] / (ccc[0] + 0.1)
ccc.reset_index(inplace=True)
ft.df_change_colname(ccc, {0: "sex_0", 1: "sex_1"})
tmp_train_user = pd.merge(tmp_train_user, ccc, on=['sex'],
                          how='left')  # train_user_info行维度 与 train_target相同，不需要再groupby

# In[]:
# 7.1.2.2、account_grade会员级别：
# 将性别的np.nan更换为“未知”
ft.missValue_fillna(tmp_train_user, "account_grade", "未知")  # 将np.nan作为 分类特征 的一种类别，则无需进行 列向（每一行）缺失值比例 统计。
# In[]:
# 交叉表： account_grade 与 target 的关系
ccc = pd.crosstab(tmp_train_user["account_grade"], tmp_train_user["target"])
ccc['account_grade_default_rate'] = ccc[1] / (ccc[0] + 0.1)
ccc.reset_index(inplace=True)
ft.df_change_colname(ccc, {0: "account_grade_0", 1: "account_grade_1"})
tmp_train_user = pd.merge(tmp_train_user, ccc, on=['account_grade'], how='left')

# In[]:
# 7.1.2.3、how_old： 申请贷款时间 - 生日
tmp_train_user['how_old'] = tmp_train_user.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days / 365, axis=1)

# In[]:
# 7.2、train_target表 连接 train_user表的 新特征
train_target_user_NewF = tmp_train_user.copy()
# In[]:
# 删除多余特征：
'''
1、生成交叉表的0、1列
2、sex列（有sex违约得分特征了）
3、account_grade列（有account_grade违约得分特征了）
'''
temp_columns = ["sex", "sex_0", "sex_1", "account_grade", "account_grade_0", "account_grade_1"]
train_target_user_NewF.drop(columns=temp_columns, inplace=True)
# In[]:
# 'industry','degree','merriage','id_card','income','hobby' 6个特征缺失值百分比90%以上，
# 且target==1 和 target==0时变化不大，做 缺失值特征标识 没有多大效果。 删除特征。
temp_columns = ['industry', 'degree', 'merriage', 'id_card', 'income', 'hobby']
train_target_user_NewF.drop(columns=temp_columns, inplace=True)
# In[]:
# 最后统一填充缺失值： 需要先统计新增特征：列向（每一行）缺失值比例； 而后再填充缺失值。
# ft.missValue_all_fillna(train_target_user_NewF, ["how_old"], 0)


# In[]:
# 合并 以及 基本特征
train_data = pd.merge(train_target, train_auth, on=['id'], how='left')
# In[]:
train_data = pd.merge(train_data, train_user, on=['id'], how='left')
# In[]:
train_data = pd.merge(train_data, train_credit, on=['id'], how='left')
# In[]:
train_data['loan_hour'] = train_data['appl_sbm_tm'].map(lambda x: x.hour)
train_data['loan_day'] = train_data['appl_sbm_tm'].map(lambda x: x.day)
train_data['loan_month'] = train_data['appl_sbm_tm'].map(lambda x: x.month)
train_data['loan_year'] = train_data['appl_sbm_tm'].map(lambda x: x.year)
train_data['nan_num'] = train_data.isnull().sum(axis=1)
# In[]:
# train_data['diff_day'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days, axis=1)
# train_data['how_old'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days/365, axis=1)
# In[]:
# train_data['是否认证时间在借贷时间前'] = train_data.apply(lambda x:0 if (x['is_auth_time_authtable'] == 0) else ( 1 if x['auth_time'] < x['appl_sbm_tm'] else 0),axis=1)
# train_data['是否认证时间在借贷时间后'] = train_data.apply(lambda x:0 if (x['is_auth_time_authtable'] == 0) else ( 1 if x['auth_time'] > x['appl_sbm_tm'] else 0),axis=1)
# train_data['认证时间在借贷时间前多少天'] = train_data.apply(lambda x:0 if (x['是否认证时间在借贷时间前'] == 0) else (x['appl_sbm_tm'] - x['auth_time']).days,axis=1)
# train_data['认证时间在借贷时间后多少天'] = train_data.apply(lambda x:0 if (x['是否认证时间在借贷时间后'] == 0) else (x['auth_time'] - x['appl_sbm_tm']).days,axis=1)
# In[]:
# train_data = pd.merge(train_data, bank_name_setlen,on=['id'],how='left')
# train_data = pd.merge(train_data, bank_num_len,on=['id'],how='left')
# train_data = pd.merge(train_data, bank_phone_num_setlen,on=['id'],how='left')
# train_data = pd.merge(train_data, bank_card_type_score,on=['id'],how='left')

# In[]:
# train_data = pd.merge(train_data,before_appl_sbm_tm_howmany,on=['id'],how='left')
# train_data = pd.merge(train_data,after_appl_sbm_tm_howmany,on=['id'],how='left')
# train_data = pd.merge(train_data,before_appl_sbm_tm_money_mean,on=['id'],how='left')
# train_data = pd.merge(train_data,after_appl_sbm_tm_money_mean,on=['id'],how='left')
# train_data = pd.merge(train_data,before_appl_sbm_tm_money_max,on=['id'],how='left')
# train_data = pd.merge(train_data,before_appl_sbm_tm_money_min,on=['id'],how='left')
# train_data = pd.merge(train_data,after_appl_sbm_tm_money_max,on=['id'],how='left')
# train_data = pd.merge(train_data,after_appl_sbm_tm_money_min,on=['id'],how='left')


# In[]:
# 连接 银行违约率临时表
# train_data = pd.merge(train_data, bank_name_score_mean, on=['id'], how='left')


# In[]:
# ft.writeFile_outData(train_data, r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\2_DataFountain\2_Immediate-AI-Global-Challenger-Competition\MyLearn\train_data.csv", "GBK")


# In[]:
IS_OFFLine = False
# In[]:
train_data = train_data.fillna(0)
# In[]:
# if IS_OFFLine == False:
#    train_data = train_data[train_data.appl_sbm_tm >= datetime.datetime(2017,1,1)]
#    train_data = train_data.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
#    print(train_data.shape)
#
# if IS_OFFLine == True:
#    dummy_fea = ['sex', 'qq_bound', 'wechat_bound','account_grade']
#    dummy_df = pd.get_dummies(train_data.loc[:,dummy_fea])
#    train_data_copy = pd.concat([train_data,dummy_df],axis=1)
#    train_data_copy = train_data_copy.fillna(0)
#    vaild_train_data = train_data_copy.drop(dummy_fea,axis=1)
#    valid_train_train = vaild_train_data[vaild_train_data.appl_sbm_tm < datetime.datetime(2017,4,1)]
#    valid_train_test = vaild_train_data[vaild_train_data.appl_sbm_tm >= datetime.datetime(2017,4,1)]
#    valid_train_train = valid_train_train.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
#    valid_train_test = valid_train_test.drop(['appl_sbm_tm','id','auth_time','phone','birthday','hobby','id_card'],axis=1)
#    vaild_train_x = valid_train_train.drop(['target'],axis=1)
#    vaild_test_x = valid_train_test.drop(['target'],axis=1)
#    redict_result = xgb_feature(vaild_train_x,valid_train_train['target'].values,vaild_test_x,None)
#    print('valid auc',roc_auc_score(valid_train_test['target'].values,redict_result))
#    sys.exit(23)

# In[]:
dummy_fea = ['sex', 'qq_bound', 'wechat_bound', 'account_grade']
dummy_df = pd.get_dummies(train_data.loc[:, dummy_fea])
train_data_copy = pd.concat([train_data, dummy_df], axis=1)
# In[]:
vaild_train_data = train_data_copy.drop(dummy_fea, axis=1)
# In[]:
valid_train_train = vaild_train_data[vaild_train_data["appl_sbm_tm"] < datetime.datetime(2017, 4, 1)]
valid_train_test = vaild_train_data[vaild_train_data["appl_sbm_tm"] >= datetime.datetime(2017, 4, 1)]
# In[]:
valid_train_train = valid_train_train.drop(['appl_sbm_tm', 'id', 'auth_time', 'phone', 'birthday', 'id_card'], axis=1)
valid_train_test = valid_train_test.drop(['appl_sbm_tm', 'id', 'auth_time', 'phone', 'birthday', 'id_card'], axis=1)

# In[]:
vaild_train_x = valid_train_train.drop(['target'], axis=1)
vaild_train_y = valid_train_train['target'].values

vaild_test_x = valid_train_test.drop(['target'], axis=1)
vaild_test_y = valid_train_test['target'].values
# In[]:
redict_result, model = xgb_feature(vaild_train_x, vaild_train_y, vaild_test_x, None)
print('valid auc', roc_auc_score(vaild_test_y, redict_result))  # valid auc 0.8241570019871227

# In[]:
import joblib

# 同样可以看看模型被保存到了哪里
joblib.dump(model,
            r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\2_DataFountain\2_Immediate-AI-Global-Challenger-Competition\MyLearn\train_data.dat")
# In[]:
loaded_model = joblib.load(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\2_DataFountain\2_Immediate-AI-Global-Challenger-Competition\MyLearn\train_data.dat")
# In[]:
dvali = xgb.DMatrix(vaild_test_x)
ypreds = loaded_model.predict(dvali)
print('valid auc', roc_auc_score(vaild_test_y, ypreds))  # valid auc 0.8241570019871227
# In[]:
minmin = min(ypreds)
maxmax = max(ypreds)
vfunc = np.vectorize(lambda x: (x - minmin) / (maxmax - minmin))
print(vfunc(ypreds))


