# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:13:45 2019

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
import pickle

# In[]:
data = load_boston()
X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)

# In[]:
dtrain = xgb.DMatrix(Xtrain,Ytrain)

#设定参数，对模型进行训练
param = {'silent':True
          ,'obj':'reg:linear'
          ,"subsample":1
          ,"eta":0.05
          ,"gamma":20
          ,"lambda":3.5
          ,"alpha":0.2
          ,"max_depth":4
          ,"colsample_bytree":0.4
          ,"colsample_bylevel":0.6
          ,"colsample_bynode":1}
num_round = 180

bst = xgb.train(param, dtrain, num_round)

# In[]:
# 1.1、保存模型：pickle
pickle.dump(bst, open(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\10_XGBoost\xgboostonboston.dat","wb"))

#注意，open中我们往往使用w或者r作为读取的模式，但其实w与r只能用于文本文件 - txt
#当我们希望导入的不是文本文件，而是模型本身的时候，我们使用"wb"和"rb"作为读取的模式
#其中wb表示以二进制写入，rb表示以二进制读入，使用open进行保存的这个文件中是一个可以进行读取或者调用的模型

# In[]:
# 看看模型被保存到了哪里？
import sys
sys.path


# In[]:
# 1.2、数据导入： （可以清除所有数据缓存做测试）
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pickle
import xgboost as xgb

data = load_boston()

X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)

# In[]:
# 注意，如果我们保存的模型是xgboost库中建立的模型，则导入的数据类型也必须是xgboost库中的数据类型
dtest = xgb.DMatrix(Xtest,Ytest)

# In[]:
# 导入模型
loaded_model = pickle.load(open(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\10_XGBoost\xgboostonboston.dat", "rb"))
print("Loaded model from: xgboostonboston.dat")
# In[]:
# 做预测，直接调用接口predict
ypreds = loaded_model.predict(dtest)
MSE(Ytest,ypreds)
r2_score(Ytest,ypreds)



# In[]:
# 2.1、保存模型：joblib
import joblib

# 同样可以看看模型被保存到了哪里
joblib.dump(bst,r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\10_XGBoost\xgboost-boston.dat")


# In[]:
# 2.2、数据导入： （可以清除所有数据缓存做测试）
data = load_boston()

X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)

# In[]:
# 注意，如果我们保存的模型是xgboost库中建立的模型，则导入的数据类型也必须是xgboost库中的数据类型
dtest = xgb.DMatrix(Xtest,Ytest)

# In[]:
loaded_model = joblib.load(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\10_XGBoost\xgboost-boston.dat")
# In[]:
# 做预测，直接调用接口predict
ypreds = loaded_model.predict(dtest)
MSE(Ytest,ypreds)
r2_score(Ytest,ypreds)



# In[]:
# 使用sklearn中的模型
bst = XGBR(n_estimators=200
           ,eta=0.05,gamma=20
           ,reg_lambda=3.5
           ,reg_alpha=0.2
           ,max_depth=4
           ,colsample_bytree=0.4
           ,colsample_bylevel=0.6).fit(Xtrain,Ytrain) #训练完毕

# In[]:
joblib.dump(bst,r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\10_XGBoost\xgboost-boston-sklearn.dat")
# In[]:
loaded_model = joblib.load(r"E:\soft\Anaconda\Anaconda_Python3.6_code\data_analysis\101_Sklearn\10_XGBoost\xgboost-boston-sklearn.dat")
# In[]:
# 则这里可以直接导入Xtest,直接是我们的numpy
ypreds = loaded_model.predict(Xtest)
MSE(Ytest,ypreds)
r2_score(Ytest,ypreds)





