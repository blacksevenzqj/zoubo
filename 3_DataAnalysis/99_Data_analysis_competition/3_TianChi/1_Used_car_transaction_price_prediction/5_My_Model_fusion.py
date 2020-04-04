# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:50:59 2020

@author: dell
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

import itertools
import matplotlib.gridspec as gridspec
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA
from sklearn.metrics import mean_squared_error as MSE, r2_score, mean_absolute_error as MAE

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR
import xgboost as xgb

from math import isnan
import FeatureTools as ft
ft.set_file_path(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\100_Data_analysis_competition\3_TianChi\1_Used_car_transaction_price_prediction\data")
import Tools_customize as tc
import Binning_tools as bt
import StackingModels as sm

# In[]:
train_data_6 = ft.readFile_inputData('train_data_6.csv', index_col=0) 
test_data_6 = ft.readFile_inputData('test_data_6.csv', index_col=0)
# In[]:
train_data_6 = train_data_6.fillna(-1)
test_data_6 = test_data_6.fillna(-1)
# In[]:
temp_train_miss =  ft.missing_values_table(train_data_6)
temp_test_miss =  ft.missing_values_table(test_data_6)

# In[]:
print(train_data_6.dtypes)
ft.reduce_mem_usage(train_data_6, False)
train_data_6.dtypes
# In[]:
print(test_data_6.dtypes)
ft.reduce_mem_usage(test_data_6, False)
test_data_6.dtypes

# In[]:
feature_names = train_data_6.columns.tolist()
feature_names.remove('price')
X_data = train_data_6[feature_names]
Y_data = train_data_6['price']
X_test = test_data_6[feature_names]



# In[]:
# 1.1、xgb原生库： obj：默认binary:logistic
# 使用类Dmatrix读取数据
dtrain = xgb.DMatrix(X_data, Y_data)
dtest = xgb.DMatrix(X_test)

# 写明参数，silent默认为False，通常需要手动设置为True，将它关闭。
# 原生库的 silent默认为False，打印日志；  Sklearn的 silent默认为False，打印日志（一般手动设置为True，不打印）
param = {'silent':False,'objective':'reg:squarederror',"eta":0.13,"gamma":20} # 和前面的推测相同，不应该加subsample参数。
num_round = 250 # 多少次迭代/多少颗树 相当于 Sklearn的XGB中的n_estimators
#类 train，可以直接导入的参数是训练数据，树的数量，其他参数都需要通过params来导入
bst = xgb.train(param, dtrain, num_round)
# In[]:
# 接口predict
print(r2_score(Y_data, bst.predict(dtrain))) # 0.8946301885375523
print(MSE(Y_data, bst.predict(dtrain))) # 0.0020066018
print(MAE(Y_data, bst.predict(dtrain))) # 0.03179782

# In[]:
predict_result = bst.predict(dtest)
predict_result = pd.DataFrame(predict_result, columns=['predict'])
# In[]:
# 测试还原 price：
train_data_4_min_price = 2.3978952727983707
train_data_4_max_price = 10.676669748432332

predict_result['predict_minmax'] = predict_result['predict'] * (train_data_4_max_price - train_data_4_min_price) + train_data_4_min_price
predict_result['predict_minmax_log'] = np.exp(predict_result['predict_minmax'])
predict_result['predict_final'] = np.round(predict_result['predict_minmax_log'])

# In[]:
# 1.2、Sklearn库：
bst_skl = XGBR(n_estimators=250, random_state=420, silent=True, objective="reg:squarederror", learning_rate=0.13, gamma=20)
bst_skl.fit(X_data, Y_data)
# In[]:
print(r2_score(Y_data, bst_skl.predict(X_data))) # 0.8946348682692177
print(MSE(Y_data, bst_skl.predict(X_data))) # 0.00200635456280437
print(MAE(Y_data, bst_skl.predict(X_data))) # 0.031795600421283834
# In[]:
predict_result_skl = bst_skl.predict(X_test)
predict_result_skl = pd.DataFrame(predict_result, columns=['predict'])



# In[]:
# My Stacking：
# 模型融合中使用到的各个单模型
clfs = [XGBR(n_estimators=250, random_state=420, silent=True, objective="reg:squarederror", learning_rate=0.13, gamma=20)
        , LGBMR(num_leaves=63,n_estimators=100,learning_rate=0.1)
#        , GradientBoostingRegressor(loss='ls',subsample=0.85,max_depth=5,n_estimators=100,learning_rate=0.1) # 不能处理np.nan
        ]
# In[]:
# 调参测试 Stacking模型：
sm.Stacking_Regressor_customize(clfs, X_data, Y_data)
# In[]:



