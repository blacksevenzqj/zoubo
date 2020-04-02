# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:32:17 2020

@author: dell
"""

import pandas as pd
import datetime
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score as CVS, train_test_split as TTS
from xgboost import XGBRegressor as XGBR
import xgboost as xgb
import re
from sklearn.metrics import roc_auc_score, mean_absolute_error,  make_scorer
from sklearn.metrics import mean_squared_error as MSE, r2_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
#import pandas_profiling
color = sns.color_palette()
sns.set_style('darkgrid')

from math import isnan
import FeatureTools as ft
ft.set_file_path(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\3_TianChi\1_Used_car_transaction_price_prediction\data")
import Tools_customize as tc
import Binning_tools as bt

# In[]:
train_data_5 = ft.readFile_inputData('train_data_5.csv', index_col=0) # price是大于0的
# In[]:
temp_data_miss =  ft.missing_values_table(train_data_5)

# In[]:
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'city', 'kilometer', 'power_cut_bin', 'diff_day_cut_bin']
temp_col = ['kilometer', 'power_cut_bin', 'diff_day_cut_bin']
categorical_astype_str_col = ft.set_diff(categorical_features, temp_col)[1] # 差集： 27-7=20
# In[]:
# 1、特征类型转换
#for i in categorical_astype_str_col:
#    ft.num_to_char(train_data_5, i)
#    
#train_data_5.dtypes
# In[]:
# 特征类型转换 以 减少内存消耗
ft.reduce_mem_usage(train_data_5, False)

train_data_5.dtypes

# In[]:
temp_data_miss2 =  ft.missing_values_table(train_data_5)
# In[]:
# 先删除 notRepairedDamage 的 np.nan （检测下效果）
temp_data_miss_, train_data_6 = ft.missing_values_table(train_data_5, percent=16, del_type=2)
# In[]:
temp_data_miss_ = ft.missing_values_table(train_data_6)
# In[]:
# 接着 删除 特征中包含 np.nan 的行
temp_data_miss_, train_data_6 = ft.missing_values_table(train_data_6, percent=0, del_type=2)
# In[]:

# In[]:
feature_names = train_data_6.columns.tolist() # 36
numeric_features = ft.set_diff(feature_names, categorical_features)[1] # 差集： 37-8=26
numeric_features.remove("price")
feature_names.remove("price")

# In[]:
# 2、方差分析（分类特征 的 回归分析）
from sklearn.feature_selection import f_regression

# 特征不能有np.nan
F, pvalues_f = f_regression(train_data_6[categorical_features], train_data_6['price'])
k = F.shape[0] - (pvalues_f > 0.05).sum()

#X_fsF = SelectKBest(f_regression, k=392).fit_transform(X_fsvar, y)
#cross_val_score(RFC(n_estimators=10,random_state=0),X_fsF,y,cv=5).mean()
# In[]:
# city的pvalues_f > 0.05，删除吧
temp_f_regression = [*zip(categorical_features, pvalues_f)]
# In[]:
categorical_features.remove('city')
feature_names.remove("city")
# In[]:
train_data_6.drop('city', axis=1, inplace=True)



# In[]:
train_data_6.dtypes

# In[]:
# 3、简单模型： 
train_X = train_data_6[feature_names]
train_y = train_data_6['price']
# In[]:
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)
model = model.fit(train_X, train_y)
# In[]:
print('intercept:'+ str(model.intercept_)) # 截距
print(tc.list_expand_tuple(feature_names, model.coef_))

# In[]:
subsample_index = tc.get_randint(low=0, high=len(train_y), size=50)
# In[]:
# 绘制特征v_9的值与标签的散点图，图片发现模型的预测结果（蓝色点）与真实标签（黑色点）的分布
# 预测值 与 真实值 的 散点分布： X轴为连续特征， Y轴为预测值/真实值
ft.feature_predic_actual_scatter(train_X, train_y, 'v_9', 'price', model)

# In[]:
# 异方差 不明显
ft.heteroscedastic_singe(train_X, train_y, 'v_9', 'price', False)

# In[]:
# 线性回归模型（线性回归、岭回归）： 方差与泛化误差 学习曲线： Ridge回归直接不能用 R2一直下降
ft.linear_model_comparison(train_X, train_y, cv_customize=5, start=1, end=201, step=50)

# In[]:
# 交叉验证
scores = ft.make_scorer_metrics_cv(model, train_X, train_y, is_log_transfer=False)
np.mean(scores)

# In[]:
# 基于样本量学习曲线
ft.plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y[:1000], 'neg_mean_squared_error')



# In[]:
# Embedded嵌入法： 因变量Y必须是int类型： CJBD
'''
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt

# 随机森林实例化： （不能有np.nan： 是随机森林 还是 模型选择SelectFromModel 的要求？）
RFC_ = RFC(n_estimators=100, random_state=0)
# 筛选特征（数据）： 针对 树模型的 feature_importances_ 属性删选： 1/35 = 0.02857， 因变量Y必须是int类型，否则报错，这不是扯么？？？
X_embedded_1 = SelectFromModel(RFC_, threshold=0.02857).fit_transform(train_X.values, train_y.astype(int).values)
# In[]:
sfmf = SelectFromModel(RFC_,threshold=0.02857).fit(train_X, train_y.astype(int).values)
X_embedded_2_index = sfmf.get_support(indices=True) # 特征选择后 特征的 原列位置索引
X_embedded_2 = sfmf.transform(train_X)
print(train_X.columns[X_embedded_2_index]) # 特征选择后 特征的 原列名称索引
# 在这里我只想取出来有限的特征。0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，因为平均每个特征
# 只能够分到大约0.001 = 1/780 的feature_importances_
#模型的维度明显被降低了
'''


# In[]:
# XGBoost：
# 一、选 n_estimators： 
# Sklearn库
# 1.1、样本量学习曲线： 检测过拟合情况 （一折交叉验证，机器顶不住）
cv = ShuffleSplit(n_splits=1, test_size=.2, random_state=0)

ft.plot_learning_curve(XGBR(n_estimators=100,random_state=420,silent=True,objective="reg:squarederror")
                    ,"XGB",train_X,train_y,ax=None,cv=cv)
plt.show()

# In[]
# 1.2、方差与泛化误差 学习曲线：  （一折交叉验证，机器顶不住）
cv = ShuffleSplit(n_splits=1, test_size=.2, random_state=0)

axisx = range(100,300,50)
ft.learning_curve_r2_customize(axisx, train_X, train_y, cv)
'''
R2最大值时对应的n_estimators参数取值:250.000000； R2最大值:0.947486； R^2最大值对应的R^2方差值:0.000000
R2方差最小值时对应的n_estimators参数取值:100.000000； R2方差最小值对应的R2值:0.940488； R2方差最小值:0.000000
泛化误差可控部分最小值时对应的n_estimators参数取值:250.000000； 泛化误差可控部分最小值时对应的R2值:0.947486； 泛化误差可控部分最小值时对应的R2方差值:0.000000； 泛化误差可控部分最小值:0.002758
'''
# 选 n_estimators参数取值:250
# In[]:
# 1.3、再次以选出的n_estimators最优参数 运行 样本量学习曲线
cv = ShuffleSplit(n_splits=1, test_size=.2, random_state=0)

ft.plot_learning_curve(XGBR(n_estimators=250, random_state=420, silent=True)
                    ,"XGB",train_X,train_y,ax=None,cv=cv)
plt.show()



# In[]:
# xgb原生库： 
# 2.1、评估指标要么在param的map中指定（非xgboost.cv函数）； 2、要么直接在xgb.cv函数中指定，不能一起指定。
param1 = {'silent':True,'obj':'reg:squarederror',"gamma":0}  # "eval_metric":"rmse"，默认rmse
#param2 = {'silent':True,'obj':'reg:squarederror',"gamma":20}
num_round = 250
n_fold=3 # 必须最少3折交叉验证
# 回归模型：默认均方误差
ft.learning_curve_xgboost(train_X, train_y, param1, None, num_round, "rmse", n_fold, None, set_ylim_top=0.2) # 默认rmse
# 训练集并未平稳，还的继续调参，机器顶不住了，暂时就到这吧

# In[]:
# 2.2、最终调参方式： 3组参数/3个模型 在一个图中显示 进行评估指标对比调参
# 这里就摆个示例： 还有 2组参数/2个模型，机器顶不住了
dfull = xgb.DMatrix(train_X,train_y)

# 默认超参数：
param1 = {'silent':True # 默认False： 打印
          ,'obj':'reg:linear' # 默认分类： binary:logistic
          ,"subsample":1 # 默认1
          ,"max_depth":6 # 默认6
          ,"eta":0.3 # 默认0.3
          ,"gamma":0 # 默认0
          ,"lambda":1 # 默认1，L2正则
          ,"alpha":0 # 默认0，L1正则
          ,"colsample_bytree":1
          ,"colsample_bylevel":1
          ,"colsample_bynode":1
          }
num_round = 250

# 必须最少3折交叉验证
cvresult1 = xgb.cv(param1, dfull, num_boost_round=num_round, metrics=("rmse"), nfold=3)

fig,ax = plt.subplots(1,figsize=(15,8))
ax.set_ylim(top=1)
ax.grid()
ax.plot(range(1,num_round+1),cvresult1.iloc[:,2],c="red",label="train,original")
ax.plot(range(1,num_round+1),cvresult1.iloc[:,0],c="orangered",label="test,original")


# In[]:

# In[]:

# In[]:

# In[]:

# In[]:

# In[]:

