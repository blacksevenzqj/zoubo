# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:08:56 2019

@author: dell
"""

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

# In[]:
data = load_boston()
X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)

# In[]:
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain) # 训练
y_predict = reg.predict(Xtest)
print(reg.score(Xtest,Ytest)) # R^2评估指标  0.9197580267581366
print(np.mean(y), MSE(Ytest, y_predict), MSE(Ytest, y_predict)/np.mean(y))
# In[]:
# 树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法(SelectFromModel)进行特征选择
temparr = reg.feature_importances_

# In[]:
# 交叉验证：
reg = XGBR(n_estimators=100)

# 不严谨： 全数据集（如果数据量少，就用全数据集方式）
CVS(reg,X,y,cv=5).mean() # R^2
# In[]:
# 严谨： 分训练和测试
CVS(reg,Xtrain,Ytrain,cv=5).mean() # R^2， scoring='neg_mean_squared_error'

# In[]:
import sklearn
sorted(sklearn.metrics.SCORERS.keys())

# In[]:
# 使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=100)
CVS(rfr,Xtrain,Ytrain,cv=5).mean() # 0.7975497480638329

# In[]:
# silent：打印日志
#如果开启参数slient：在数据巨大，预料到算法运行会非常缓慢的时候可以使用这个参数来监控模型的训练进度
# xgboost库 默认是silent=False会打印训练进程，设置silent=True不会打印训练进程，只返回运行结果。
reg = XGBR(n_estimators=10,silent=False)
# sklearn库中的xgbsoost的默认为silent=True不会打印训练进程，想打印需要手动设置为False
CVS(reg,Xtrain,Ytrain,cv=5,scoring='neg_mean_squared_error').mean()#-92.67865836936579



# In[]:
# A、集成算法框架超参数：
# 学习曲线： 
# 一、基于样本量(交叉验证学习曲线函数)
import FeatureTools as ft

#cv = KFold(n_splits=5, shuffle = True, random_state=42) #交叉验证模式
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

ft.plot_learning_curve(XGBR(n_estimators=100,random_state=420)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
'''
样本量阈值[ 28  91 155 219 283]
交叉验证训练集阈值283,最大分数0.853370
前提：Xtrain=354，cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)，那么CVtrian=354*0.8=283；已确定的n_estimators=100
就是说 当交叉验证训练集样本量达到最大值283时 和 交叉验证训练集阈值283 相同，模型R^2最大0.853370
这证明了：正常来说样本量越大，模型才不容易过拟合，效果越好。
'''

# In[]:
# 二、基于超参数（按顺序 依次确定 超参数）
# 1、n_estimators：
'''
axisx = range(10,1010,50)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs)) # 560 0.863497886588919
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
'''
# 训练样本Xtrain=404个，交叉验证后最佳分数0.8634 需要n_estimators=560颗树，是很奇怪的（她的数据切分方式结果）。
# In[]:
# 1.1、细化学习曲线
#cv = KFold(n_splits=5, shuffle = True, random_state=42) 
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

axisx = range(100,300,10)
ft.learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv)
'''
270 0.8628488903325771 0.0010233348954168013
170 0.8590591457326957 0.0009745514733593459
270 0.8628488903325771 0.0010233348954168013 0.019833761778422276
最后选择 n_estimators=270 的超参数
'''
# In[]:
# 1.2、再次以选出的最优参数 运行 样本量学习曲线
ft.plot_learning_curve(XGBR(n_estimators=270, random_state=420)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
'''
样本量阈值[ 28  91 155 219 283]
交叉验证训练集阈值283,最大分数0.862849（较第一次n_estimators=100的样本量学习曲线R^2上升）
前提：Xtrain=354，cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)，那么CVtrian=354*0.8=283；已确定的n_estimators=270。
就是说 当交叉验证训练集样本量达到最大值283时 和 交叉验证训练集阈值283 相同，模型R^2最大0.853370，较第一次n_estimators=100的样本量学习曲线R^2上升。 
这再次证明了：1、正常来说样本量越大，模型才不容易过拟合，效果越好。2.1、n_estimators参数控制树的数量决定了模型的学习能力，树的数量越多，模型的学习能力越强，
2.2、树的数量提升对模型的影响有极限，最开始，模型的表现会随着XGB的树的数量一起提升，但到达某个点之后，树的数量越多，模型的效果会逐步下降，这也说明了暴力增加n_estimators不一定有效果。
'''
# In[]:
# 1.3、验证模型效果
time0 = time()
print(XGBR(n_estimators=100,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)

time0 = time()
print(XGBR(n_estimators=560,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest)) 
print(time()-time0)

# 可以看到，最后选择的超参数n_estimators=270，R2得分最高0.9216，且耗时也较小0.1147
time0 = time()
print(XGBR(n_estimators=270,random_state=420).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)


# In[]:
# 2、subsample： （已确定n_estimators=270）
axisx = np.linspace(0,1,20)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=270,subsample=i,random_state=420)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="green",label="XGB")
plt.legend()
plt.show()

# In[]:
# 2.1、细化学习曲线
axisx = np.linspace(0.05,1,20)
ft.learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, hparam_name="subsample", prev_hparam_value=[270])
# In[]:
# 2.2、细化学习曲线
axisx = np.linspace(0.7,1,25)
ft.learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, hparam_name="subsample", prev_hparam_value=[270])
'''
0.75 0.8762683263611173 0.0004238536122880797
0.75 0.8762683263611173 0.0004238536122880797
0.75 0.8762683263611173 0.0004238536122880797 0.01573338067376706
subsample=0.75时，R^2最高0.8762683263611173，R^2方差最小0.0004238536122880797，泛化误差可控部分0.01573338067376706，那么CVtrian*subsample = 212个样本
我们的模型现在正处于样本量过少并且过拟合的状态，根据学习曲线展现出来的规律，我们的训练样本量在200左右的时候，模型的效果有可能反而比更多训练数据的时候好，但这不代表模型的泛化能力在更小的训练样本量下会更强。
正常来说样本量越大，模型才不容易过拟合，现在展现出来的效果，是由于我们的样本量太小造成的一个巧合。从这个角度来看，我们的subsample参数对模型的影响应该会非常不稳定，大概率应该是无法提升模型的泛化能力的，但也不乏提升模型的可能性。
'''
# In[]:
# 2.3、再次以选出的最优参数 运行 样本量学习曲线
ft.plot_learning_curve(XGBR(n_estimators=270,subsample=0.75, random_state=420)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
'''
样本量阈值[ 28  91 155 219 283]
交叉验证训练集阈值283,最大分数0.876268（是之前 样本量学习曲线 中R^2最高的）
前提：Xtrain=354，cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)，那么CVtrian=354*0.8=283；已确定的n_estimators=270，subsample=0.75。
就是说 当交叉验证训练集样本量达到最大值283时 和 交叉验证训练集阈值283 相同（实际交叉验证训练集样本量：CVtrian*subsample = 212），模型R^2最大0.876268，是之前 样本量学习曲线 中R^2最高的。
这再次证明了：1、正常来说样本量越大，模型才不容易过拟合，效果越好。2.1、n_estimators参数控制树的数量决定了模型的学习能力，树的数量越多，模型的学习能力越强，
2.2、树的数量提升对模型的影响有极限，最开始，模型的表现会随着XGB的树的数量一起提升，但到达某个点之后，树的数量越多，模型的效果会逐步下降，这也说明了暴力增加n_estimators不一定有效果。
3、subsample参数在小样本量情况下对模型的影响大概率应该是无法提升模型的泛化能力的，但也不乏提升模型的可能性。
'''
# In[]:
# 2.4、验证模型效果
reg = XGBR(n_estimators=270
           ,subsample=0.75
           ,random_state=420).fit(Xtrain,Ytrain)

print(reg.score(Xtest,Ytest)) # 0.9252577873218268  R^2提升了
print(MSE(Ytest,reg.predict(Xtest))) # 6.955053266305905
'''
虽然 超参数n_estimators=270和subsample=0.75都使模型R^2提升了，但在同一个测试集上R^2的提升，并不一定表示 模型的泛化能力 提升。
且由于我们的数据集过少，在小样本量情况下降低抽样的比例subsample超参数反而让数据的效果更低（正常来说），正因如此，保持默认subsample=1。
'''


# In[]:
# 3、eta（迭代决策树）
'''
xgb.train() XGBoost原生库
eta：默认0.3
取值范围[0,1]

xgb.XGBRegressor() SKLearn库
learning_rate：默认0.1
取值范围[0,1]
'''

# 3.1、由于 eta迭代次数 和 n_estimators 超参数密切相关，需要一起搜索，所以使用GridSearchCV。 
"""
# 运行时间2分多钟
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':np.arange(100,300,10),
    'learning_rate':np.arange(0.05,1,0.05) 
}
reg = XGBR(random_state=420)
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, verbose=1, cv=cv, scoring='r2') # neg_mean_squared_error
grid_search.fit(Xtrain, Ytrain)
'''
reg:linear is now deprecated in favor of reg:squarederror.
现在不推荐使用reg：linear，而推荐使用reg：squarederror。
XGBoost的重要超参数objective损失函数选项： reg：linear → reg：squarederror
'''

print(grid_search.best_score_) # 0.870023216964111
print(grid_search.best_params_) # {'learning_rate': 0.25, 'n_estimators': 260}
best_reg = grid_search.best_estimator_ # 最佳分类器
print(best_reg)
testScore = best_reg.score(Xtest,Ytest)
print("GridSearchCV测试结果：", testScore) # 0.8996310370746 分数比 学习曲线的低。。。
# [Parallel(n_jobs=1)]: Done 1900 out of 1900 | elapsed:  2.1min finished

# 后使用neg_mean_squared_error评价指标，和R^2结果相同。
"""
# In[]:
# 3.2、再画学习曲线：
axisx = np.linspace(0.01,0.3,30)
ft.learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, hparam_name="learning_rate", prev_hparam_value=[270,0.75])
'''
0.12999999999999998 0.8829549083764519 0.0002829219197039983
0.16999999999999998 0.8626896941963793 0.00014794913715299236
0.12999999999999998 0.8829549083764519 0.0002829219197039983 0.01398247539286878
'''
# In[]:
# 3.3、验证模型效果
reg = XGBR(n_estimators=270
            ,subsample=0.75
            ,learning_rate=0.13
            ,random_state=420).fit(Xtrain,Ytrain)

print(reg.score(Xtest,Ytest)) # 0.9302325654075598  R^2提升了
print(MSE(Ytest,reg.predict(Xtest))) # 6.492130838208879
