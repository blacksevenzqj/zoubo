# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:09:31 2019

@author: dell
"""

import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
import FeatureTools as ft

# In[]:
data = load_boston()
X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

# In[]:
# B、弱评估器超参数：
# 4、booster（选择弱评估器）
for booster in ["gbtree", "gblinear", "dart"]:
    reg = XGBR(n_estimators=260
               , learning_rate=0.25
               , random_state=420
               , booster=booster
               , silent=True).fit(Xtrain, Ytrain)
    print(booster)
    print(reg.score(Xtest, Ytest))

# gblinear线性弱评估器表现最差： 说明 波斯顿数据集 不是线性数据集（特征X 与 因变量Y 不是线性联系）

# In[]:
# 5、objective（损失函数）
# Sklearn的XGB： objective：默认reg:linear
reg = XGBR(n_estimators=270, subsample=0.75, learning_rate=0.13, random_state=420).fit(Xtrain, Ytrain)
print(reg.score(Xtest, Ytest))
print(MSE(Ytest, reg.predict(Xtest)))
# In[]:
# xgb原生库： obj：默认binary:logistic
# 使用类Dmatrix读取数据
dtrain = xgb.DMatrix(Xtrain, Ytrain)
dtest = xgb.DMatrix(Xtest, Ytest)
# 非常遗憾无法打开来查看，所以通常都是先读到pandas里面查看之后再放到DMatrix中
dtrain

# 写明参数，silent默认为False，通常需要手动设置为True，将它关闭。
# 原生库的 silent默认为False，打印日志；  Sklearn的 silent默认为False，打印日志（一般手动设置为True，不打印）
param = {'silent': False, 'objective': 'reg:linear', "eta": 0.13}  # 和前面的推测相同，不应该加subsample参数。
num_round = 270  # 多少次迭代/多少颗树 相当于 Sklearn的XGB中的n_estimators
# 类 train，可以直接导入的参数是训练数据，树的数量，其他参数都需要通过params来导入
bst = xgb.train(param, dtrain, num_round)

# 接口predict
print(r2_score(Ytest, bst.predict(dtest)))
print(MSE(Ytest, bst.predict(dtest)))

# In[]:
# 6、正则化参数1： alpha，lambda（回归模型）
# 使用网格搜索来查找最佳的参数组合（运行20分钟，了解即可）
# cv = KFold(n_splits=5, shuffle = True, random_state=42)
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

'''
reg = XGBR(n_estimators=270,subsample=0.75,learning_rate=0.13,random_state=420)
param = {"reg_alpha":np.arange(0,5,0.05),"reg_lambda":np.arange(0,2,0.05)}
gscv = GridSearchCV(reg,param_grid = param,scoring = "neg_mean_squared_error",cv=cv)
#======【TIME WARNING：10~20 mins】======#
time0=time()
gscv.fit(Xtrain,Ytrain)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
gscv.best_params_
gscv.best_score_
preds = gscv.predict(Xtest)

r2_score(Ytest,preds)
MSE(Ytest,preds)
'''

# In[]:
# 7、正则化参数2： gamma
# 7.1、SkLearn库XGBoost模型
'''
axisx = np.arange(0,5,0.05)
运行速度较缓慢并且曲线的效果匪夷所思。全无法从中看出什么趋势，偏差时高时低，方差时大时小，
参数γ引起的波动远远超过其他参数（其他参数至少还有一个先上升再平稳的过程，而γ则是仿佛完全无规律）。
在sklearn下XGBoost太不稳定，如果这样来调整参数的话，效果就很难保证。
（那是你把gamma范围调的太小了，但最好还是使用XGBoost原生库）
'''
axisx = np.arange(10, 31, 1)  # 这个gamma取值范围的计算结果 和 XGBoost原生库 差不多
ft.learning_curve_r2_customize(axisx, X, y, cv, hparam_name="gamma", prev_hparam_value=[270, 1, 0.1])

# In[]:
# 7.2、自定义交叉验证（XGBoost原生库）
ss = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)  # 意思就是不重复刷新打乱数据
axisx = np.arange(10, 31, 1)
param_fixed = {'silent': True, 'obj': 'reg:linear', "eval_metric": "rmse"}  # 默认rmse
num_round = 270

ft.learning_curve_xgboost_customize(axisx, X, y, ss, param_fixed, "gamma", num_round)

# In[]:
# 7.3、xgboost原生交叉验证类： xgboost.cv
'''
gamma是如何控制过拟合？ 
1、控制训练集上的训练：即，降低训练集上的表现（R^2降低、MSE升高），从而使训练集表现 和 测试集的表现 逐步趋近。
2、gamma不断增大，训练集R^2降低、MSE升高，训练集表现 和 测试集的表现 逐步趋近；但随着gamma不断增大，测试集也会出现R^2降低、MSE升高 的 欠拟合情况。所以，需要找到gamma的平衡点。
3、gamma主要是用来 降低模型复杂度、提高模型泛化能力的（防止过拟合）；不是用来提高模型准确性的（降低欠拟合）。
'''
# 回归例子：
param1 = {'silent': True, 'obj': 'reg:linear', "gamma": 0}  # 默认rmse
param2 = {'silent': True, 'obj': 'reg:linear', "gamma": 20}
num_round = 270
n_fold = 5
# 回归模型：默认均方误差
ft.learning_curve_xgboost(X, y, param1, param2, num_round, "rmse", n_fold)  # 默认rmse

# In[]:
# 分类例子：
from sklearn.datasets import load_breast_cancer

data2 = load_breast_cancer()
x2 = data2.data
y2 = data2.target

param1 = {'silent': True, 'obj': 'binary:logistic', "gamma": 0}
param2 = {'silent': True, 'obj': 'binary:logistic', "gamma": 0.8}
num_round = 100
n_fold = 5
# 分类模型： 默认错误率
ft.learning_curve_xgboost(x2, y2, param1, param2, num_round, "error", n_fold)  # 默认error







