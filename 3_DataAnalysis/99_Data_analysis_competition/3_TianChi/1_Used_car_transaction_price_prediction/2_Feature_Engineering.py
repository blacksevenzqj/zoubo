# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:10:21 2020

@author: dell
"""

'''
SaleID	交易ID，唯一编码
name	汽车交易名称，已脱敏
regDate	汽车注册日期，例如20160101，2016年01月01日
model	车型编码，已脱敏
brand	汽车品牌，已脱敏
bodyType	车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7
fuelType	燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6
gearbox	变速箱：手动：0，自动：1
power	发动机功率：范围 [ 0, 600 ]
kilometer	汽车已行驶公里，单位万km
notRepairedDamage	汽车有尚未修复的损坏：是：0，否：1
regionCode	地区编码，已脱敏
seller	销售方：个体：0，非个体：1
offerType	报价类型：提供：0，请求：1
creatDate	汽车上线时间，即开始售卖时间
price	二手车交易价格（预测目标）
v系列特征	匿名特征，包含v0-14在内15个匿名特征
'''
# In[]:
import pandas as pd
import datetime
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import xgboost as xgb
import re
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# import pandas_profiling
color = sns.color_palette()
sns.set_style('darkgrid')

from math import isnan
import FeatureTools as ft

ft.set_file_path(
    r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\3_TianChi\1_Used_car_transaction_price_prediction\data")
import Tools_customize as tc
import Binning_tools as bt

# In[]:
# 一、表读取
train_data = ft.readFile_inputData('used_car_train_20200313.csv', parse_dates=['regDate', 'creatDate'], sep=' ')
test_data = ft.readFile_inputData('used_car_testA_20200313.csv', parse_dates=['regDate', 'creatDate'], sep=' ')
# In[]:
print(train_data.shape, test_data.shape)  # (150000, 31) (50000, 30)
print(train_data[train_data['price'] <= 0].shape)

# In[]
# 1、缺失值
train_miss = ft.missing_values_table(train_data)
test_miss = ft.missing_values_table(test_data)

# In[]:
train_data_1 = train_data.copy()
test_data_1 = test_data.copy()

# In[]:
# 2、异常值检测
# 2.1、power发动机功率：
box_more_index_power, hist_more_index_power = ft.outlier_detection(train_data_1, 'power', train_data_1, 'price',
                                                                   fit_type=3, box_scale=3)
# box_more_index_power_test, hist_more_index_power_test = ft.outlier_detection(test_data_1, 'power', box_scale=3)
# In[]:
print(len(box_more_index_power[0]), len(box_more_index_power[1]), len(hist_more_index_power[0]),
      len(hist_more_index_power[1]))
# print(len(box_more_index_power_test[0]), len(box_more_index_power_test[1]), len(hist_more_index_power_test[0]), len(hist_more_index_power_test[1]))
# In[]:
# 删5倍标准差以外的数据
train_data_1.drop(hist_more_index_power[0], axis=0, inplace=True)
train_data_1.drop(hist_more_index_power[1], axis=0, inplace=True)
# test_data_1.drop(hist_more_index_power_test[0], axis=0, inplace=True)
# test_data_1.drop(hist_more_index_power_test[1], axis=0, inplace=True)
ft.recovery_index([train_data_1])

# In[]:
# 2.2、price价格：
box_more_index_price, hist_more_index_price = ft.outlier_detection(train_data_1, 'price', box_scale=3)
# In[]:
print(len(box_more_index_price[0]), len(box_more_index_price[1]), len(hist_more_index_price[0]),
      len(hist_more_index_price[1]))
# In[]:
# 删5倍标准差以外的数据
train_data_1.drop(hist_more_index_price[0], axis=0, inplace=True)
train_data_1.drop(hist_more_index_price[1], axis=0, inplace=True)
ft.recovery_index([train_data_1])

# In[]:
# 2.3、'creatDate', 'regDate'： creatDate - regDate 得到使用时间天数
# 使用时间：data['creatDate'] - data['regDate']，反应汽车使用时间，一般来说价格与使用时间成反比
# 不过要注意，数据里有时间出错的格式，所以我们需要 errors='coerce'
ft.custom_time_conversion(train_data_1, 'regDate', 'creatDate')
train_data_1.drop(['creatDate', 'regDate', 'regDate1', 'regDate2', 'regDate3', 'diff_day1', 'diff_day2'], axis=1,
                  inplace=True)
# In[]:
# 这两个特征 是按照 它的方法计算的： 所以有pd.lib.NaT值
# 看一下空数据，有 15k 个样本的时间是有问题的，我们可以选择删除，也可以选择放着。
# 但是这里不建议删除，因为删除缺失数据占总样本量过大，7.5%
# 我们可以先放着，因为如果我们 XGBoost 之类的决策树，其本身就能处理缺失值，所以可以不用管；
# print(train_data_1['diff_day1'].isnull().sum()) # 15057 它的做法 减法后 有空值
# print(train_data_1['regDate1'].isnull().sum())
# In[]:
ft.custom_time_conversion(test_data_1, 'regDate', 'creatDate')
test_data_1.drop(['creatDate', 'regDate', 'regDate1', 'regDate2', 'regDate3', 'diff_day1', 'diff_day2'], axis=1,
                 inplace=True)

# In[]:
# 2.4、regionCode邮政编码： 从邮编中提取城市信息，因为是德国的数据，所以参考德国的邮编，相当于加入了先验知识
train_data_1['city'] = train_data_1['regionCode'].map(
    lambda x: str(x)[:-3] if len(str(x)) == 4 else str(x)[:-2] if len(str(x)) == 3 else str(x)[:-1] if len(
        str(x)) == 2 else str(x))
print(train_data_1['city'].isnull().sum())
train_data_1[['regionCode', 'city']][0:10]
train_data_1.drop('regionCode', axis=1, inplace=True)
# In[]
test_data_1['city'] = test_data_1['regionCode'].map(
    lambda x: str(x)[:-3] if len(str(x)) == 4 else str(x)[:-2] if len(str(x)) == 3 else str(x)[:-1] if len(
        str(x)) == 2 else str(x))
print(test_data_1['city'].isnull().sum())
test_data_1[['regionCode', 'city']][0:10]
test_data_1.drop('regionCode', axis=1, inplace=True)

# In[]
# 1、缺失值
train_1_miss = ft.missing_values_table(train_data_1)
test_1_miss = ft.missing_values_table(test_data_1)

# In[]
# 2.5、测试自动删除指定 缺失值比例的特征列行数据
mis_val_table_ren_columns, df_nm = ft.missing_values_table(train_data_1, customize_axis=0, percent=0, del_type=2)
# In[]:
# 删除 model特征中 只有1个缺失值的 一行数据
train_data_1.drop(train_data_1.loc[train_data_1['model'].isnull()].index, axis=0, inplace=True)

# In[]:
# 2.5、notRepairedDamage汽车有尚未修复的损坏 缺失值类别转换
train_data_1['notRepairedDamage'] = train_data_1['notRepairedDamage'].map(lambda x: np.nan if x == '-' else float(x))
train_data_1['notRepairedDamage'].value_counts()
# In[]:
test_data_1['notRepairedDamage'] = test_data_1['notRepairedDamage'].map(lambda x: np.nan if x == '-' else float(x))
test_data_1['notRepairedDamage'].value_counts()

# In[]:
# 2.6、seller分类特征： 类别严重偏斜 0:149999 1:1 删除该特征 （根本不用看测试集了）
print(train_data_1['seller'].value_counts())
train_data_1.drop('seller', axis=1, inplace=True)
test_data_1.drop('seller', axis=1, inplace=True)
# In[]:
# 2.7、offerType分类特征： 类别严重偏斜 0:150000 删除该特征 （根本不用看测试集了）
print(train_data_1['offerType'].value_counts())
train_data_1.drop('offerType', axis=1, inplace=True)
test_data_1.drop('offerType', axis=1, inplace=True)

# In[]:
train_data_2 = train_data_1.copy()
test_data_2 = test_data_1.copy()

# In[]:
# 2.8、计算汽车品牌的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['brand'])
# In[]:
train_gb = train_data_2.groupby("brand")
all_info = {}
for kind, kind_data in train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
# all_info字典 转 DataFrame： 字典的key就是DataFrame的行索引index，所以要 .T置转 → .reset_index()重置行索引 → .rename(columns={"index": "brand"})跟换列名： 将all_info的key的原行索引名index（置转后现在列名）更新名称
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})

# In[]:
# 2.9、车身类型的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['bodyType'])
# In[]:
temp_data = train_data_2[train_data_2['price'] > 0]  # 数据 条件 提前做好
agg = {'bodyType_amount': len, "bodyType_price_max": np.max, "bodyType_price_median": np.median,
       "bodyType_price_min": np.min, "bodyType_price_sum": np.sum, "bodyType_price_std": np.std,
       "bodyType_price_average": np.mean}
bodyType_fe = tc.groupby_agg_oneCol(temp_data, ['bodyType'], 'price', agg, as_index=False)

# In[]:
# 2.10、燃油类型的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['fuelType'])
# In[]:
temp_data = train_data_2[train_data_2['price'] > 0]  # 数据 条件 提前做好
agg = {'fuelType_amount': len, "fuelType_price_max": np.max, "fuelType_price_median": np.median,
       "fuelType_price_min": np.min, "fuelType_price_sum": np.sum, "fuelType_price_std": np.std,
       "fuelType_price_average": np.mean}
fuelType_fe = tc.groupby_agg_oneCol(temp_data, ['fuelType'], 'price', agg, as_index=False)

# In[]:
# 2.11、变速箱类型的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['gearbox'])
# In[]:
temp_data = train_data_2[train_data_2['price'] > 0]  # 数据 条件 提前做好
agg = {'gearbox_amount': len, "gearbox_price_max": np.max, "gearbox_price_median": np.median,
       "gearbox_price_min": np.min, "gearbox_price_sum": np.sum, "gearbox_price_std": np.std,
       "gearbox_price_average": np.mean}
gearbox_fe = tc.groupby_agg_oneCol(temp_data, ['gearbox'], 'price', agg, as_index=False)

# In[]:
# 2.12、汽车已行驶公里类型的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['kilometer'])
# In[]:
temp_data = train_data_2[train_data_2['price'] > 0]  # 数据 条件 提前做好
agg = {'kilometer_amount': len, "kilometer_price_max": np.max, "kilometer_price_median": np.median,
       "kilometer_price_min": np.min, "kilometer_price_sum": np.sum, "kilometer_price_std": np.std,
       "kilometer_price_average": np.mean}
kilometer_fe = tc.groupby_agg_oneCol(temp_data, ['kilometer'], 'price', agg, as_index=False)

# In[]:
# 数据分桶 以 power 为例
# 这时候我们的缺失值也进桶了，
# 为什么要做数据分桶呢，原因有很多，= =
# 1. 离散后稀疏向量内积乘法运算速度更快，计算结果也方便存储，容易扩展；
# 2. 离散后的特征对异常值更具鲁棒性，如 age>30 为 1 否则为 0，对于年龄为 200 的也不会对模型造成很大的干扰；
# 3. LR 属于广义线性模型，表达能力有限，经过离散化后，每个变量有单独的权重，这相当于引入了非线性，能够提升模型的表达能力，加大拟合；
# 4. 离散后特征可以进行特征交叉，提升表达能力，由 M+N 个变量编程 M*N 个变量，进一步引入非线形，提升了表达能力；
# 5. 特征离散后模型更稳定，如用户年龄区间，不会因为用户年龄长了一岁就变化

# 当然还有很多原因，LightGBM 在改进 XGBoost 时就增加了数据分桶，增强了模型的泛化性
# In[]:
# 2.13、斯皮尔曼自动分箱： power发动机功率
data_spearmanr_power, cut_updown_power, bin_num_power, r_list_power = bt.spearmanr_auto_bins(train_data_2['power'],
                                                                                             train_data_2['price'])
# In[]:
unique_label_power, counts_label_power, unique_dict_power = ft.category_quantity_statistics_all(
    data_spearmanr_power['Bucket'])
# In[]:
ft.box_diagram(data_spearmanr_power, 'Bucket', 'price', is_violin=False)
# In[]:
ft.box_diagram(data_spearmanr_power, 'Bucket', 'price', is_violin=True)
# In[]:
cut_updown_power[0], cut_updown_power[-1] = -np.inf, np.inf
# In[]:
train_data_2['power_cut_bin'] = pd.cut(train_data_2['power'], cut_updown_power, retbins=False)
# In[]:
# 分类变量转换：
le = LabelEncoder()
le.fit(train_data_2['power_cut_bin'])
train_data_2['power_cut_bin'] = le.transform(train_data_2['power_cut_bin'])
# In[]:
# 发动机功率类型的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['power_cut_bin'])
# In[]:
# 新增特征
temp_data = train_data_2[train_data_2['price'] > 0]  # 数据 条件 提前做好
agg = {'power_cut_bin_amount': len, "power_cut_bin_price_max": np.max, "power_cut_bin_price_median": np.median,
       "power_cut_bin_price_min": np.min, "power_cut_bin_price_sum": np.sum, "power_cut_bin_price_std": np.std,
       "power_cut_bin_price_average": np.mean}
power_cut_bin_fe = tc.groupby_agg_oneCol(temp_data, ['power_cut_bin'], 'price', agg, as_index=False)

# In[]:
# test数据集
test_data_2["power_cut_bin"] = pd.cut(test_data_2['power'], cut_updown_power, retbins=False)  # 可以指定labels=[0,1,2,3]参数
test_data_2['power_cut_bin'] = le.transform(test_data_2['power_cut_bin'])
ft.category_quantity_statistics_value_counts(test_data_2, ['power_cut_bin'])

# In[]:
# 2.14、斯皮尔曼自动分箱： diff_day3使用时间（天数）
data_spearmanr_diff_day, cut_updown_diff_day, bin_num_diff_day, r_list_diff_day = bt.spearmanr_auto_bins(
    train_data_2['diff_day3'], train_data_2['price'])
# In[]:
unique_label_diff_day, counts_label_diff_day, unique_dict_diff_day = ft.category_quantity_statistics_all(
    data_spearmanr_diff_day['Bucket'])
# In[]:
ft.box_diagram(data_spearmanr_diff_day, 'Bucket', 'price', is_violin=False)
# In[]:
ft.box_diagram(data_spearmanr_diff_day, 'Bucket', 'price', is_violin=True)
# In[]:
cut_updown_diff_day[0], cut_updown_diff_day[-1] = -np.inf, np.inf
# In[]:
train_data_2['diff_day_cut_bin'] = pd.cut(train_data_2['diff_day3'], cut_updown_diff_day, retbins=False)
# In[]:
# 分类变量转换：
le = LabelEncoder()
le.fit(train_data_2['diff_day_cut_bin'])
train_data_2['diff_day_cut_bin'] = le.transform(train_data_2['diff_day_cut_bin'])
# In[]:
# 发动机功率类型的销售统计量
ft.category_quantity_statistics_value_counts(train_data_2, ['diff_day_cut_bin'])
# In[]:
# 新增特征
temp_data = train_data_2[train_data_2['price'] > 0]  # 数据 条件 提前做好
agg = {'diff_day_cut_bin_amount': len, "diff_day_cut_bin_price_max": np.max, "diff_day_cut_bin_price_median": np.median,
       "diff_day_cut_bin_price_min": np.min, "diff_day_cut_bin_price_sum": np.sum, "diff_day_cut_bin_price_std": np.std,
       "diff_day_cut_bin_price_average": np.mean}
diff_day_cut_bin_fe = tc.groupby_agg_oneCol(temp_data, ['diff_day_cut_bin'], 'price', agg, as_index=False)

# In[]:
# test数据集
test_data_2["diff_day_cut_bin"] = pd.cut(test_data_2['diff_day3'], cut_updown_diff_day,
                                         retbins=False)  # 可以指定labels=[0,1,2,3]参数
test_data_2['diff_day_cut_bin'] = le.transform(test_data_2['diff_day_cut_bin'])
ft.category_quantity_statistics_value_counts(test_data_2, ['diff_day_cut_bin'])

# In[]:
train_data_3 = train_data_2.copy()
test_data_3 = test_data_2.copy()

# In[]:
# 合并特征：
train_data_3 = train_data_3.merge(brand_fe, how='left', on='brand')
test_data_3 = test_data_3.merge(brand_fe, how='left', on='brand')
# In[]:
train_data_3 = train_data_3.merge(bodyType_fe, how='left', on='bodyType')
test_data_3 = test_data_3.merge(bodyType_fe, how='left', on='bodyType')
# In[]:
train_data_3 = train_data_3.merge(fuelType_fe, how='left', on='fuelType')
test_data_3 = test_data_3.merge(fuelType_fe, how='left', on='fuelType')
# In[]:
train_data_3 = train_data_3.merge(gearbox_fe, how='left', on='gearbox')
test_data_3 = test_data_3.merge(gearbox_fe, how='left', on='gearbox')
# In[]:
train_data_3 = train_data_3.merge(kilometer_fe, how='left', on='kilometer')
test_data_3 = test_data_3.merge(kilometer_fe, how='left', on='kilometer')
# In[]:
train_data_3 = train_data_3.merge(power_cut_bin_fe, how='left', on='power_cut_bin')
train_data_3.drop('power', axis=1, inplace=True)
test_data_3 = test_data_3.merge(power_cut_bin_fe, how='left', on='power_cut_bin')
test_data_3.drop('power', axis=1, inplace=True)
# In[]:
train_data_3 = train_data_3.merge(diff_day_cut_bin_fe, how='left', on='diff_day_cut_bin')
train_data_3.drop('diff_day3', axis=1, inplace=True)
test_data_3 = test_data_3.merge(diff_day_cut_bin_fe, how='left', on='diff_day_cut_bin')
test_data_3.drop('diff_day3', axis=1, inplace=True)

# In[]:
'''
['SaleID', 'name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox',
       'kilometer', 'notRepairedDamage', 'price', 'v_0', 'v_1', 'v_2', 'v_3',
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
       'v_13', 'v_14', 'city', 'power_cut_bin', 'diff_day_cut_bin',
       'brand_amount', 'brand_price_average', 'brand_price_max',
       'brand_price_median', 'brand_price_min', 'brand_price_std',
       'brand_price_sum', 'bodyType_amount', 'bodyType_price_max',
       'bodyType_price_median', 'bodyType_price_min', 'bodyType_price_sum',
       'bodyType_price_std', 'bodyType_price_average', 'fuelType_amount',
       'fuelType_price_max', 'fuelType_price_median', 'fuelType_price_min',
       'fuelType_price_sum', 'fuelType_price_std', 'fuelType_price_average',
       'gearbox_amount', 'gearbox_price_max', 'gearbox_price_median',
       'gearbox_price_min', 'gearbox_price_sum', 'gearbox_price_std',
       'gearbox_price_average', 'kilometer_amount', 'kilometer_price_max',
       'kilometer_price_median', 'kilometer_price_min', 'kilometer_price_sum',
       'kilometer_price_std', 'kilometer_price_average',
       'power_cut_bin_amount', 'power_cut_bin_price_max',
       'power_cut_bin_price_median', 'power_cut_bin_price_min',
       'power_cut_bin_price_sum', 'power_cut_bin_price_std',
       'power_cut_bin_price_average', 'diff_day_cut_bin_amount',
       'diff_day_cut_bin_price_max', 'diff_day_cut_bin_price_median',
       'diff_day_cut_bin_price_min', 'diff_day_cut_bin_price_sum',
       'diff_day_cut_bin_price_std', 'diff_day_cut_bin_price_average']
'''
train_data_3.columns
# In[]:
'''
['SaleID', 'name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox',
       'power', 'kilometer', 'notRepairedDamage', 'v_0', 'v_1', 'v_2', 'v_3',
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
       'v_13', 'v_14', 'diff_day3', 'city']
'''
test_data_3.columns
# In[]:
# 导出保存：
ft.writeFile_outData(train_data_3, "train_data_3.csv")
ft.writeFile_outData(test_data_3, "test_data_3.csv")
# train_data_3 = ft.readFile_inputData('train_data_3.csv', index_col=0) # price是大于0的
# test_data_3 = ft.readFile_inputData('test_data_3.csv', index_col=0)
# In[]:
numeric_features = [
    'price',

    'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
    'v_11', 'v_12', 'v_13', 'v_14',

    'brand_amount', 'brand_price_average', 'brand_price_max',
    'brand_price_median', 'brand_price_min', 'brand_price_std', 'brand_price_sum',

    'bodyType_amount', 'bodyType_price_max', 'bodyType_price_median',
    'bodyType_price_min', 'bodyType_price_sum', 'bodyType_price_std', 'bodyType_price_average',

    'fuelType_amount', 'fuelType_price_max', 'fuelType_price_median',
    'fuelType_price_min', 'fuelType_price_sum', 'fuelType_price_std', 'fuelType_price_average',

    'gearbox_amount', 'gearbox_price_max', 'gearbox_price_median',
    'gearbox_price_min', 'gearbox_price_sum', 'gearbox_price_std', 'gearbox_price_average',

    'kilometer_amount', 'kilometer_price_max', 'kilometer_price_median',
    'kilometer_price_min', 'kilometer_price_sum', 'kilometer_price_std', 'kilometer_price_average',

    'power_cut_bin_amount', 'power_cut_bin_price_max', 'power_cut_bin_price_median',
    'power_cut_bin_price_min', 'power_cut_bin_price_sum', 'power_cut_bin_price_std', 'power_cut_bin_price_average',

    'diff_day_cut_bin_amount', 'diff_day_cut_bin_price_max', 'diff_day_cut_bin_price_median',
    'diff_day_cut_bin_price_min', 'diff_day_cut_bin_price_sum', 'diff_day_cut_bin_price_std',
    'diff_day_cut_bin_price_average']

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'city',
                        'kilometer', 'power_cut_bin', 'diff_day_cut_bin']

# In[]:
# 3、连续特征偏度
# 所有连续变量都求一遍：
fig, axe = plt.subplots(2, 1, figsize=(120, 16))
skew, kurt, var_x_ln = ft.skew_distribution_test(train_data_3[numeric_features], axe)
# In[]:
# 偏度>1的再求一遍：
than_one_columns = var_x_ln.tolist()
fig, axe = plt.subplots(2, 1, figsize=(100, 16))
skew, kurt, var_x_ln = ft.skew_distribution_test(train_data_3[than_one_columns], axe)
# In[]:
'''
than_one_columns = ['fuelType_price_max',
 'brand_price_min',
 'fuelType_price_min',
 'fuelType_price_sum',
 'v_7',
 'v_2',
 'v_5',
 'kilometer_price_min',
 'kilometer_price_max',
 'v_11',
 'brand_price_median',
 'bodyType_price_min',
 'price',
 'brand_price_max',
 'bodyType_price_median',
 'kilometer_price_median',
 'kilometer_price_average',
 'diff_day_cut_bin_price_min',
 'power_cut_bin_price_max',
 'v_0',
 'gearbox_price_median',
 'gearbox_price_average',
 'gearbox_price_max',
 'gearbox_price_sum',
 'gearbox_amount',
 'gearbox_price_std',
 'brand_price_average',
 'kilometer_price_std',
 'v_14',
 'bodyType_price_max',
 'power_cut_bin_price_median',
 'diff_day_cut_bin_price_median',
 'diff_day_cut_bin_price_max',
 'diff_day_cut_bin_price_average',
 'diff_day_cut_bin_price_sum',
 'diff_day_cut_bin_price_std']
'''
print(len(than_one_columns))  # 36
than_one_columns

# In[]:
f, axes = plt.subplots(1, 2, figsize=(23, 8))
ft.con_data_distribution(train_data_3, 'fuelType_price_max', axes, fit_type=1, box_scale=3)
# In[]:
# 取对数再看：
ft.logarithm(train_data_3, ['fuelType_price_max'], f_type=1)
# In[]:
f, axes = plt.subplots(1, 2, figsize=(23, 8))
ft.con_data_distribution(train_data_3, 'fuelType_price_max_ln', axes, fit_type=1, box_scale=3)
# In[]:
# ss = StandardScaler() # 理论取值范围是(-∞,+∞)，但经验上看大多数取值范围在[-4,4]之间
mm = MinMaxScaler()  # 数据会完全落入[0,1]区间内（z-score没有类似区间）
# 标准化再看：
train_data_3['fuelType_price_max_1'] = mm.fit_transform(train_data_3[['fuelType_price_max_ln']])  # iris.data
# In[]:
f, axes = plt.subplots(1, 2, figsize=(23, 8))
ft.con_data_distribution(train_data_3, 'fuelType_price_max_1', axes, fit_type=1, box_scale=3)
# In[]:
# 先删除测试特征
train_data_3.drop(['fuelType_price_max_ln', 'fuelType_price_max_1'], axis=1, inplace=True)

# In[]:
# 每个偏度>1的特征查 手动 看一遍：
f, axes = plt.subplots(1, 2, figsize=(23, 8))
ft.con_data_distribution(train_data_3, 'brand_price_average', axes, fit_type=1, box_scale=3)

# In[]:
del_skew_columns = ['fuelType_price_max', 'brand_price_min', 'fuelType_price_min', 'fuelType_price_sum',
                    'kilometer_price_min', 'kilometer_price_max']
# train_data_3.drop(del_skew_columns, axis=1, inplace=True)
# test_data_3.drop(del_skew_columns, axis=1, inplace=True)

# In[]:
# 偏度>1的连续特征  -  手动需删除的偏度>1的连续特征  =  剩下的偏度>1的连续特征
temp_list = ft.set_diff(than_one_columns, del_skew_columns)
stay_columns = temp_list[1]  # 差集
# In[]:
# '''
stay_columns = ['bodyType_price_max',
                'bodyType_price_median',
                'bodyType_price_min',
                'brand_price_average',
                'brand_price_max',
                'brand_price_median',
                'diff_day_cut_bin_price_average',
                'diff_day_cut_bin_price_max',
                'diff_day_cut_bin_price_median',
                'diff_day_cut_bin_price_min',
                'diff_day_cut_bin_price_std',
                'diff_day_cut_bin_price_sum',
                'gearbox_amount',
                'gearbox_price_average',
                'gearbox_price_max',
                'gearbox_price_median',
                'gearbox_price_std',
                'gearbox_price_sum',
                'kilometer_price_average',
                'kilometer_price_median',
                'kilometer_price_std',
                'power_cut_bin_price_max',
                'power_cut_bin_price_median',
                'price',
                'v_0',
                'v_11',
                'v_14',
                'v_2',
                'v_5',
                'v_7']
# '''
print(len(stay_columns))  # 30
stay_columns.sort()
stay_columns

# In[]:
train_data_4 = train_data_3.copy()
test_data_4 = test_data_3.copy()
# In[]:
# 导出保存：
# ft.writeFile_outData(train_data_4, "train_data_4.csv")
# ft.writeFile_outData(test_data_4, "test_data_4.csv")
train_data_4 = ft.readFile_inputData('train_data_4.csv', index_col=0)
test_data_4 = ft.readFile_inputData('test_data_4.csv', index_col=0)
print(train_data_4[train_data_4['price'] <= 0].shape)
# In[]:
# log转换： 只能转一次哦
# stay_columns为： 偏度>1的连续特征 - 手动需删除的偏度>1的连续特征 = 剩下的偏度>1的连续特征。 所以才取log：
ft.logarithm(train_data_4, stay_columns, f_type=2)
print(train_data_4[train_data_4['price'] <= 0].shape)
ft.logarithm(test_data_4, ft.set_diff(stay_columns, ['price'])[1], f_type=2)
# In[]:
# 再看偏度
fig, axe = plt.subplots(2, 1, figsize=(120, 16))
skew, kurt, var_x_ln = ft.skew_distribution_test(train_data_4[stay_columns], axe)
# In[]:
f, axes = plt.subplots(1, 2, figsize=(23, 8))
ft.con_data_distribution(train_data_4, 'gearbox_price_median', axes, fit_type=1, box_scale=3)

# In[]:
# 4、归一化：
# 经过 偏度筛选（偏度>1的都取log） 剩下的 所有连续特征
numeric_skew_stay_cols = ft.set_diff(numeric_features, del_skew_columns)[1]  # 差集 59
numeric_skew_stay_cols.sort()
# In[]:
# 经过偏度处理剩下的 所有连续特征 标准化， 包括 因变量Y
# ss = StandardScaler() # 理论取值范围是(-∞,+∞)，但经验上看大多数取值范围在[-4,4]之间
mm = MinMaxScaler()  # 数据会完全落入[0,1]区间内（z-score没有类似区间）
mm.fit(train_data_4[numeric_skew_stay_cols])
train_data_4[numeric_skew_stay_cols] = mm.transform(train_data_4[numeric_skew_stay_cols])
print(train_data_4[train_data_4['price'] <= 0].shape)  # 需要删除的
# In[]:
ft.simple_drop_data(train_data_4, train_data_4[train_data_4['price'] <= 0].index)
print(train_data_4[train_data_4['price'] <= 0].shape)  # 需要删除的
# In[]
numeric_skew_stay_cols_no_price = ft.set_diff(numeric_skew_stay_cols, ['price'])[1]
test_data_4[numeric_skew_stay_cols_no_price] = mm.fit_transform(test_data_4[numeric_skew_stay_cols_no_price])
# In[]:
# 为还原 price 准备：
train_data_4_min_price = np.min(train_data_4['price'])  # 2.3978952727983707
train_data_4_max_price = np.max(train_data_4['price'])  # 10.676669748432332
# In[]:
# 测试还原 price：
train_data_4['price_1'] = train_data_4['price'] * (
        train_data_4_max_price - train_data_4_min_price) + train_data_4_min_price
print(np.exp(train_data_4['price_1']))
train_data_4.drop('price_1', axis=1, inplace=True)

# In[]:
# 5、特征选择
# 5.1、过滤法：
# 5.1.1、皮尔森相似度
temp_corr_abs_withY, temp_corr_withY = ft.corrFunction_withY(train_data_4[numeric_skew_stay_cols], 'price')

# In[]:
numeric_skew_stay_cols_not_y = ft.set_diff(numeric_skew_stay_cols, ['price'])[1]  # 差集
# In[]:
temp_corr_abs, temp_corr = ft.corrFunction(train_data_4[numeric_skew_stay_cols_not_y])
# In[]:
temp_corr_abs = temp_corr_abs[temp_corr_abs['Correlation_Coefficient'] >= 0.8]
# In[]:
ft.recovery_index(temp_corr_abs)

# In[]:
# 根据皮尔森相似度 自动特征选择
del_list, equal_list = ft.feature_select_corr_withY(train_data_4, numeric_skew_stay_cols_not_y, 'price')
# In[]:
# equal_list存在对因变量Y贡献度相等的特征： ['gearbox_price_median=gearbox_price_sum']
# 随便挑一个删除
# del_list.append('gearbox_price_median')

# In[]:
temp_data = train_data_4[numeric_skew_stay_cols_not_y].drop(del_list, axis=1)
# In[]:
# 皮尔森相似度 连续特征选择后 预留的连续特征
'''
reserve_columns = ['bodyType_price_median',
 'bodyType_price_min',
 'bodyType_price_sum',
 'brand_price_max',
 'brand_price_median',
 'brand_price_sum',
 'diff_day_cut_bin_amount',
 'diff_day_cut_bin_price_max',
 'diff_day_cut_bin_price_min',
 'fuelType_price_median',
 'gearbox_amount',
 'gearbox_price_min',
 'kilometer_price_median',
 'power_cut_bin_amount',
 'power_cut_bin_price_max',
 'power_cut_bin_price_median',
 'v_0',
 'v_10',
 'v_11',
 'v_14',
 'v_2',
 'v_3',
 'v_5',
 'v_7',
 'v_9']
'''
reserve_columns = temp_data.columns.tolist()
reserve_columns.sort()
print(len(reserve_columns))
reserve_columns  # 25

# In[]:
temp_data_miss = ft.missing_values_table(temp_data)
# 这几个特征是有np.nan的
temp_data_miss_col = temp_data_miss.index.tolist()  # 有7个特征有np.nan，所以要排除掉
reserve_columns_not_miss = ft.set_diff(reserve_columns, temp_data_miss_col)[1]  # 差集： 27-7=20
# In[]:
'''
temp_data_miss_col = ['bodyType_price_median',
 'bodyType_price_min',
 'bodyType_price_sum',
 'fuelType_price_median',
 'gearbox_amount',
 'gearbox_price_min']
'''
temp_data_miss_col.sort()
print(len(temp_data_miss_col))  # 6
temp_data_miss_col

# In[]:
'''
reserve_columns_not_miss = ['brand_price_max',
 'brand_price_median',
 'brand_price_sum',
 'diff_day_cut_bin_amount',
 'diff_day_cut_bin_price_max',
 'diff_day_cut_bin_price_min',
 'kilometer_price_median',
 'power_cut_bin_amount',
 'power_cut_bin_price_max',
 'power_cut_bin_price_median',
 'v_0',
 'v_10',
 'v_11',
 'v_14',
 'v_2',
 'v_3',
 'v_5',
 'v_7',
 'v_9']
'''
reserve_columns_not_miss.sort()
print(len(reserve_columns_not_miss))  # 19
reserve_columns_not_miss

# In[]:
# 5.1.2、方差选择： （不能有np.nan）
from sklearn.feature_selection import VarianceThreshold

# 实例化，不填参数默认方差为0
selector = VarianceThreshold()
# 获取删除不合格特征之后的新特征矩阵
X_var0 = selector.fit_transform(temp_data[reserve_columns_not_miss])  # 19
X_var0.shape  # (149166, 19) # 都保留了，没有方差为0的特征

# In[]:
# 5.1.3、互信息： （消耗大， 就不弄了）
'''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif as MIC

result = MIC(temp_data, train_data_4['price'])
k = result.shape[0] - sum(result <= 0)

X_fsmic = SelectKBest(MIC, k=392).fit_transform(temp_data, train_data_4['price'])
cross_val_score(RFC(n_estimators=10,random_state=0), X_fsmic, train_data_4['price'], cv=3).mean()
'''

# In[]:


# In[]:
# 5.2、Embedded嵌入法：（还暂时没调通。。。）
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt

# 经过 偏度筛选 → 皮尔森相似度筛选 → 方差筛选： 剩下的 连续特征
temp_X = temp_data[reserve_columns_not_miss]  # 差不多剩下26个特征（每次运行都会有不同）
y = train_data_4['price']
# In[]:
# 随机森林实例化： （不能有np.nan： 是随机森林 还是 模型选择SelectFromModel 的要求？）
RFC_ = RFC(n_estimators=10, random_state=0)
# 筛选特征（数据）： 针对 树模型的 feature_importances_ 属性删选； 因变量Y必须是int类型，否则报错，log型y就不能用，这不是扯么？？？
X_embedded_1 = SelectFromModel(RFC_, threshold=0.04).fit_transform(temp_X, y)
# In[]:
sfmf = SelectFromModel(RFC_, threshold=0.005).fit(temp_X, y)
X_embedded_2_index = sfmf.get_support(indices=True)  # 特征选择后 特征的 原列位置索引
X_embedded_2 = sfmf.transform(temp_X)
print(temp_X.columns[X_embedded_2_index])  # 特征选择后 特征的 原列名称索引
# 在这里我只想取出来有限的特征。0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，因为平均每个特征
# 只能够分到大约0.001 = 1/780 的feature_importances_
# 模型的维度明显被降低了


# In[]:
all_colunms = reserve_columns + categorical_features + ["price"]

all_colunms_no_price = reserve_columns + categorical_features
# In[]:
'''
all_colunms = ['bodyType',
 'bodyType_price_median',
 'bodyType_price_min',
 'bodyType_price_sum',
 'brand',
 'brand_price_max',
 'brand_price_median',
 'brand_price_sum',
 'city',
 'diff_day_cut_bin',
 'diff_day_cut_bin_amount',
 'diff_day_cut_bin_price_max',
 'diff_day_cut_bin_price_min',
 'fuelType',
 'fuelType_price_median',
 'gearbox',
 'gearbox_amount',
 'gearbox_price_min',
 'kilometer',
 'kilometer_price_median',
 'model',
 'name',
 'notRepairedDamage',
 'power_cut_bin',
 'power_cut_bin_amount',
 'power_cut_bin_price_max',
 'power_cut_bin_price_median',
 'price',
 'v_0',
 'v_10',
 'v_11',
 'v_14',
 'v_2',
 'v_3',
 'v_5',
 'v_7',
 'v_9']
'''
all_colunms.sort()
print(len(all_colunms))
all_colunms

# In[]:
# 导出保存：
ft.writeFile_outData(train_data_4[all_colunms], "train_data_5.csv")
ft.writeFile_outData(test_data_4[all_colunms_no_price], "test_data_5.csv")

# In[]:


# In[]:


