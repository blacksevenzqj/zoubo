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

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)

# In[]:
# B、弱评估器超参数：
# 一、观测默认超参数：
dfull = xgb.DMatrix(X,y)

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
num_round = 200

time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_boost_round=num_round, metrics=("rmse"), nfold=5)
#print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

fig,ax = plt.subplots(1,figsize=(15,8))
ax.plot(range(1,201),cvresult1.iloc[:,2],c="red",label="train,original")
ax.plot(range(1,201),cvresult1.iloc[:,0],c="orange",label="test,original")
ax.set_ylim(top=5) # 截取Y轴最大值 进行显示
ax.grid()
ax.legend(fontsize="xx-large")
plt.show()
'''
从曲线上可以看出，模型现在处于过拟合的状态。我们决定要进行剪枝。我们的目标是：训练集和测试集的结果尽量
接近，如果测试集上的结果不能上升，那训练集上的结果降下来也是不错的选择（让模型不那么具体到训练数据，增加泛化能力）。
'''

# In[]:
# 二、学习曲线调参：
'''
一、调参要求：
1、测试集上的模型指标（MSE） 较默认超参数模型 要降低（最少持平）；
2、允许训练集上的模型指标（MSE） 较默认超参数模型 升高；
3、多个模型在都满足1、2条件的情况下，选择 训练集与测试集 MSE距离近的模型（泛化误差小）
'''
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
num_round = 300

time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_boost_round=num_round, metrics=("rmse"), nfold=5)
#print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
print(time()-time0)

fig,ax = plt.subplots(1,figsize=(15,8))
ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1,num_round+1),cvresult1.iloc[:,2],c="red",label="train,original")
ax.plot(range(1,num_round+1),cvresult1.iloc[:,0],c="orangered",label="test,original")

'''
二、调参方式：
我们要使用三组曲线：
1.1、一组用于展示默认超参数的结果：param1；
1.2、一组用于展示上一个参数调节完毕后的结果：param2；
1.3、最后一组用于展示现在我们在调节的参数的结果：param3。

2.1、param2 和 param3 都从一个超参数开始调整（max_depth 或 gamma），param2先设置为默认。
2.2、每次调参都在param3中进行，将param3中确定的超参数放入到param2中； 
2.3、添加新的超参数进param3中进行调整，和param2中已确定的超参数进行对比。

三、调参顺序：
1、练习的顺序：  先确定max_depth
max_depth → eta → gamma → lambda → alpha → colsample_bytree → colsample_bylevel → colsample_bynode

2、建议的顺序：
2.1、先使用网格搜索找出比较合适的n_estimators和eta组合；
2.2、然后使用gamma或者max_depth观察模型处于什么样的状态（过拟合还是欠拟合，处于方差-偏差图像的左边还是右边？），最后再决定是否要进行剪枝；
2.3、采样subsample 和 抽样参数colsample_xxx（纵向抽样影响更大）；
2.4、最后才是正则化的两个参数：L2正则λ 和 L1正则α。

num_boost_round与eta同调 → max_depth/gamma → colsample_bytree → colsample_bylevel → colsample_bynode → subsample → lambda → alpha                        
'''

param2 = {'silent':True
          ,'obj':'reg:linear'
          ,"max_depth":2
          ,"eta":0.07
          ,"gamma":0 # 在已经设置了 max_depth 情况下，gamma默认
          ,"lambda":1.5
          ,"alpha":3
          ,"colsample_bytree":1 # 经过前面的主要超参数，本参数已不能起效，保持默认1
          ,"colsample_bylevel":0.4
          ,"colsample_bynode":0.8
          }
#param3 = {'silent':True
#          ,'obj':'reg:linear'
#          ,"max_depth":2
#          ,"eta":0.07
#          ,"gamma":0
#          ,"lambda":1.5
#          ,"alpha":3
#          ,"colsample_bytree":1
#          ,"colsample_bylevel":0.4
#          ,"colsample_bynode":0.8
#          }

# 她最后给出的 先调整gamma的结果，和之前自己调的超参数 做一个对比（貌似不行）
param3 = {'silent':True
          ,'obj':'reg:linear'
          ,"subsample":1
          ,"eta":0.05
          ,"gamma":20
          ,"lambda":3.5
          ,"alpha":0.2
          ,"max_depth":4
          ,"colsample_bytree":0.4
          ,"colsample_bylevel":0.6
          ,"colsample_bynode":1
          }

time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_boost_round=num_round, metrics=("rmse"), nfold=5)
#print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
print(time()-time0)

time0 = time()
cvresult3 = xgb.cv(param3, dfull, num_boost_round=num_round, metrics=("rmse"), nfold=5)
#print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
print(time()-time0)

ax.plot(range(1,num_round+1),cvresult2.iloc[:,2],c="blue",label="train,last")
ax.plot(range(1,num_round+1),cvresult2.iloc[:,0],c="dodgerblue",label="test,last")

ax.plot(range(1,num_round+1),cvresult3.iloc[:,2],c="m",label="train,this")
ax.plot(range(1,num_round+1),cvresult3.iloc[:,0],c="magenta",label="test,this")

ax.legend(fontsize="xx-large")
plt.show()

