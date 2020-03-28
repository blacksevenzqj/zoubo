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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
# 备份
# ft.writeFile_outData(train_data, "used_car_train_Backup.csv")
# In[]:
# train_data.head().append(train_data.tail())
# test_data.head().append(test_data.tail())
print(train_data.shape, test_data.shape)  # (150000, 31) (50000, 30)

# In[]:
# describe描述信息：
# 瞬间掌握数据的大概的范围以及每个值的异常值的判断，比如有的时候会发现999 9999 -1 等值这些其实都是nan的另外一种表达方式，有的时候需要注意下
temp1 = train_data.describe()
temp2 = test_data.describe()
# In[]:
# info来熟悉数据类型
train_data.info()
test_data.info()
# In[]:
# 1、缺失值
train_miss = ft.missing_values_table(train_data)
test_miss = ft.missing_values_table(test_data)
# In[]:
train_miss['Missing_Values'].plot.bar()
# In[]:
test_miss['Missing_Values'].plot.bar()
# In[]:
# nan存在的个数是否真的很大，如果很小一般选择填充，如果使用lgb等树模型可以直接空缺，让树自己去优化，但如果nan存在的过多、可以考虑删掉
# 可视化看下缺省值
msno.matrix(train_data.sample(250))
msno.bar(train_data.sample(1000))

# In[]:
# 2、分类特征
# 2.1、notRepairedDamage分类特征 的特殊字符“-” 转 缺失值
# 观测notRepairedDamage分类特征发现 “-” 其也可视为缺失值： “-”的个数为 24324
# 将“-”转换为np.nan之后，再次调用np.unique函数报错： 很可能是因为一个np.nan为一个类别，所以撑爆了
unique_label, counts_label, unique_dict = ft.category_quantity_statistics_all(train_data['notRepairedDamage'])
# In[]:
# 如果 输入中包含 非数字格式字符串：（PD转换数字失败） NAN、na 或 字符串“-” 其相关元素类型为<class 'str'>；
print(train_data['notRepairedDamage'].dtypes)
print(train_data.loc[0, 'notRepairedDamage'], type(train_data.loc[0, 'notRepairedDamage']))
print(train_data.loc[1, 'notRepairedDamage'], type(train_data.loc[1, 'notRepairedDamage']))
# In[]:
train_data['notRepairedDamage'] = train_data['notRepairedDamage'].map(lambda x: np.nan if x == '-' else float(x))
train_data['notRepairedDamage'].value_counts()
# In[]:
test_data['notRepairedDamage'] = test_data['notRepairedDamage'].map(lambda x: np.nan if x == '-' else float(x))
test_data['notRepairedDamage'].value_counts()
# In[]:
print(train_data['notRepairedDamage'].value_counts())  # value_counts()会剔除np.nan
print(test_data['notRepairedDamage'].value_counts())

# In[]:
# 2.2、seller分类特征： 类别严重偏斜 0:149999 1:1 删除该特征 （根本不用看测试集了）
print(train_data['seller'].value_counts())
train_data.drop('seller', axis=1, inplace=True)
test_data.drop('seller', axis=1, inplace=True)
# In[]:
# 2.3、offerType分类特征： 类别严重偏斜 0:150000 删除该特征 （根本不用看测试集了）
print(train_data['offerType'].value_counts())
train_data.drop('offerType', axis=1, inplace=True)
test_data.drop('offerType', axis=1, inplace=True)
# In[]:
# 2.4、regionCode分类特征： 地区编码，太多，保留吧
print(train_data['regionCode'].value_counts())
print(test_data['regionCode'].value_counts())
# In[]:
# 2.5、gearbox分类特征： 手动/自动，保留
# 有大量np.nan值，没有化为一个类别
print(train_data['gearbox'].value_counts())
print(test_data['gearbox'].value_counts())
# In[]:
# 2.6、bodyType分类特征： 车身类型，保留
print(train_data['bodyType'].value_counts())
print(test_data['bodyType'].value_counts())
# In[]:
# 2.7、fuelType分类特征： 燃油类型，保留
print(train_data['fuelType'].value_counts())
print(test_data['fuelType'].value_counts())

# In[]:
# 3、因变量Y分布：
# 3.1、总体分布概况（无界约翰逊分布等）
# In[]:
f, axes = plt.subplots(1, 2, figsize=(23, 6))
ft.con_data_distribution(train_data, 'price', axes)  # 正太分布
# In[]:
f, axes = plt.subplots(1, 2, figsize=(23, 6))
ft.con_data_distribution(train_data, 'price', axes, 2)  # log正太分布
# In[]:
# 价格不服从正态分布，所以在进行回归之前，它必须进行转换。虽然对数变换做得很好，但最佳拟合是无界约翰逊分布
f, axes = plt.subplots(1, 2, figsize=(23, 6))
ft.con_data_distribution(train_data, 'price', axes, 3)  # 无界约翰逊分布
# In[]:
# f = ['price', 'power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14']
fig, axe = plt.subplots(1, 1, figsize=(36, 10))
ft.normal_distribution_test(train_data, axe)  # 特征不能包含np.nan
# In[]:
# 偏度 和 峰度
# skew、kurt说明参考https://www.cnblogs.com/wyy1480/p/10474046.html
fig, axe = plt.subplots(2, 1, figsize=(30, 16))
ft.skew_distribution_test(train_data, axe)
# In[]:
# 查看因变量Y的具体频数
# 查看频数, 大于20000得值极少，其实这里也可以把这些当作特殊得值（异常值）直接用填充或者删掉，再前面进行
fig, axe = plt.subplots(1, 1, figsize=(15, 8))
ft.con_data_distribution_hist(train_data, 'price', axe)
# In[]:
# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
ft.con_data_distribution_hist(np.log(train_data['price']))
# In[]:


# In[]:
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
                    'v_11', 'v_12', 'v_13', 'v_14']
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']

# In[]:
numeric_features.append('price')
# In[]:
# 4、连续特征 相关性分析
# 4.1、皮尔森相似度
temp_corr_abs_withY, temp_corr_withY = ft.corrFunction_withY(train_data[numeric_features], 'price')
# In[]:
temp_corr_abs, temp_corr = ft.corrFunction(train_data[numeric_features])
# In[]:
temp_corr_abs = temp_corr_abs[temp_corr_abs['Correlation_Coefficient'] >= 0.8]
# In[]:
ft.recovery_index(temp_corr_abs)
# In[]:
# 根据皮尔森相似度 自动特征选择
del_list, equal_list = ft.feature_select_corr_withY(train_data, numeric_features, 'price')
# In[]:
temp_data = train_data[numeric_features].drop(del_list, axis=1)
# In[]:
# 皮尔森相似度 连续特征选择后 预留的连续特征
reserve_columns = temp_data.columns.tolist()

# In[]:
# 连续特征偏度
fig, axe = plt.subplots(2, 1, figsize=(30, 16))
ft.skew_distribution_test(train_data[reserve_columns], axe)

# In[]:
# 4.2、每个连续特征的直方图分布可视化
ft.simple_con_data_distribution(train_data, reserve_columns)
# In[]:
# 细化的直方图分布可视化
f, axes = plt.subplots(1, 2, figsize=(23, 6))
ft.con_data_distribution(train_data, 'v_10', axes)  # 正太分布

# In[]:
# 4.3、多连续特征线性关系图： （类似于皮尔森相似度的热力图）
# ft.multi_feature_linear_diagram(train_data, columns, 'price') # 太耗时，运算不了
ft.multi_feature_linear_diagram(train_data, reserve_columns)

# In[]:
Y_train = train_data['price']
# In[]:
# 此处是多变量之间的关系可视化，可视化更多学习可参考很不错的文章 https://www.jianshu.com/p/6e18d21a4cad
# 4.4、预留连续特征 与 连续因变量Y 散点分布
print(reserve_columns)
# ['power', 'kilometer', 'v_0', 'v_3', 'v_9', 'v_10', 'v_11', 'v_14', 'price']

# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# In[]:
ft.con_data_scatter(train_data, 'power', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'kilometer', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'v_0', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'v_3', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'v_9', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'v_10', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'v_11', train_data, 'price')
# In[]:
ft.con_data_scatter(train_data, 'v_14', train_data, 'price')

# In[]:
# 5、分类特征
# 5.1、分类特征nunique分布
ft.category_quantity_statistics_value_counts(train_data, categorical_features)
# In[]:
ft.category_quantity_statistics_value_counts(test_data, categorical_features)
# In[]:
# 5.2、类别特征 盒须图/小提琴图 可视化
# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
# 5.2.1、自动
categorical_features = ['model',
                        'brand',
                        'bodyType',
                        'fuelType',
                        'gearbox',
                        'notRepairedDamage']
ft.box_diagram_auto_col_category(train_data, categorical_features, 'price', is_violin=True)
# In[]:
# # 5.2.2、手动
for catg in categorical_features:
    f, axes = plt.subplots(1, 1, figsize=(10, 8))
    ft.box_diagram(train_data, catg, 'price', axes, is_violin=True)

# In[]:
# 5.3、类别特征的柱形图可视化
ft.box_diagram_auto_col_category(train_data, categorical_features, 'price', function_type=2)

# In[]:
# 5.4、类别特征的每个类别频数可视化(count_plot)
ft.box_diagram_auto_col_category(train_data, categorical_features, 'price', function_type=3)

# In[]:
# 6、用pandas_profiling生成数据报告
pfr = pandas_profiling.ProfileReport(train_data)
pfr.to_file("./example.html")

# In[]:
print(train_data['regDate'].dtypes, train_data['creatDate'].dtypes)
print(train_data.loc[0, 'regDate'], type(train_data.loc[0, 'regDate']))
print(train_data.loc[0, 'creatDate'], type(train_data.loc[0, 'creatDate']))
# In[]:
train_data['regDate1'] = train_data['regDate'].map(lambda x: pd.to_datetime(x, format="%Y%m%d"))
# ValueError: time data '20070009' does not match format '%Y%m%d' (match) 这列的日期格式有问题
# In[]:
# regDate字段包含 00月这种异常数据，怎么处理？
train_data['regDate1'] = train_data['regDate'].map(lambda x: x[4:6])
ft.category_quantity_statistics_all(train_data['regDate1'])
# In[]:
# 将是 00月的数据，截取到年。
train_data['regDate2'] = train_data['regDate'].map(lambda x: x[0:4] if x[4:6] == '00' else x)
# 并转换为PD时间格式
train_data['regDate3'] = train_data['regDate2'].map(
    lambda x: pd.to_datetime(x, format="%Y") if (len(x) == 4) else pd.to_datetime(x, format="%Y%m%d"))
# In[]:
# 新增regDate的标识列
train_data['regDate_lag'] = train_data['regDate'].map(lambda x: 0 if x[4:6] == '00' else 1)
# In[]:
# 时间差
train_data['diff_day'] = train_data.apply(lambda x: (x['creatDate'] - x['regDate3']).days, axis=1)

# In[]:
# 缺失值：行向
ft.missing_values_table(train_data)
# In[]:
# 汽车交易名称name 不是唯一行
agg = {'数量': len}
ccc = tc.groupby_agg_oneCol(train_data, ["name"], "SaleID", agg, as_index=False)
# In[]:
# 车型编码model 不是唯一行
agg = {'数量': len}
ddd = tc.groupby_agg_oneCol(train_data, ["model"], "SaleID", agg, as_index=False)

# In[]:
ccc = train_data[train_data['bodyType'].isnull()]
# In[]:
# 缺失值：列向
ft.missing_values_table(train_data, customize_axis=1)
# In[]:
# ccc = train_data[(train_data['bodyType'] == 2) & (train_data['fuelType'].isnull())]
ccc = train_data[(train_data['bodyType'] == 2)]
# In[]:
f, axes = plt.subplots(2, 2, figsize=(20, 18))
ft.class_data_distribution(ccc, 'power', 'fuelType', axes)

# In[]:
# 行驶万km分桶
ccc["kilometer_cut"], updown = pd.cut(ccc['kilometer'], [0, 5, 10, 15, 20], labels=[0, 1, 2, 3], retbins=True)
# In[]:
# 发动机功率分桶
ccc["power_cut"], updown = pd.cut(ccc['power'], [-np.inf, 100, 200, 300, 400, 500, 600, np.inf],
                                  labels=[0, 1, 2, 3, 5, 6, 7], retbins=True)


