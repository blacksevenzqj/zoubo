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
import FeatureTools as ft

# In[]:
data = load_boston()
X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)

# In[]:
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain) # 训练
y_predict = reg.predict(Xtest)
print(reg.score(Xtest,Ytest)) # R^2评估指标  0.9197580267581366
print(np.mean(y)) # 22.532806324110677
print(MSE(Ytest, y_predict)) # 7.466827353555599
print(MSE(Ytest, y_predict) / np.mean(y)) # 均方误差 大概占 y标签均值的 1/3 结果不算好

# In[]:
# 树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法(SelectFromModel)进行特征选择
temparr = reg.feature_importances_

# In[]:
# 交叉验证：
reg = XGBR(n_estimators=100)

# 不严谨： 全数据集（如果数据量少，就用全数据集方式）
CVS(reg,X,y,cv=5).mean() # R^2： cross_val_score默认 和 模型默认的评估指标相同
# In[]:
# 严谨： 分训练和测试
CVS(reg,Xtrain,Ytrain,cv=5).mean() # 默认评估指标R^2； 但可以显示指定评估指标： scoring='neg_mean_squared_error' 负均方误差

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

# L、学习曲线顺序： 基于样本量：如果过拟合（训练集、测试集相差过远） →  基于超参数：比较现实的 目标 是将训练集效果降低，从而避免过拟合 →  基于样本量：再次检测过拟合情况

# 一、基于样本量(交叉验证学习曲线函数)
# 1、线性回归测试：
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)
ft.plot_learning_curve(LinearR()
                    ,"LinearR",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
# In[]:
# 2、Sklearn的XGBT：
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

# L1、基于超参数学习曲线顺序： 确定n_estimators → 确定subsample → 确定learning_rate → 确定gamma

# 1、n_estimators：
#'''
# 前提：Xtrain=354，那么CVtrian=354*0.8=283；
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

axisx = range(10,1010,50)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420,silent=True)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs)) # 560 0.863497886588919
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
#'''
# 交叉验证训练样本CVtrian=283个，交叉验证后最佳分数0.8634 需要n_estimators=560颗树，是很奇怪的： 需要树的数量 比 训练样本量 还多
# In[]:
# 1.1、方差与泛化误差 学习曲线：
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
'''
从这个过程中观察n_estimators参数对模型的影响，我们可以得出以下结论：首先，XGB中的树的数量决定了模型的学习能力，树的数量越多，模型的学习能力越强。只要XGB中树的数量足够了，即便只有很少的数据， 模型也能够学到训练数据100%的信息，所以XGB也是天生过拟合的模型。但在这种情况下，模型会变得非常不稳定。
第二，XGB中树的数量很少的时候，对模型的影响较大，当树的数量已经很多的时候，对模型的影响比较小，只能有微弱的变化。当数据本身就处于过拟合的时候，再使用过多的树能达到的效果甚微，反而浪费计算资源。当唯一指标R^2或者准确率给出的n_estimators看起来不太可靠的时候，我们可以改造学习曲线来帮助我们。
第三，树的数量提升对模型的影响有极限，最开始，模型的表现会随着XGB的树的数量一起提升，但到达某个点之后，树的数量越多，模型的效果会逐步下降，这也说明了暴力增加n_estimators不一定有效果。
这些都和随机森林中的参数n_estimators表现出一致的状态。在随机森林中我们总是先调整n_estimators，当n_estimators的极限已达到，我们才考虑其他参数，但XGB中的状况明显更加复杂，当数据集不太寻常的时候会更加复杂。这是我们要给出的第一个超参数，因此还是建议优先调整n_estimators，一般都不会建议一个太大的数目，300以下为佳。
'''
ft.plot_learning_curve(XGBR(n_estimators=270, random_state=420, silent=True)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
'''
样本量阈值[ 28  91 155 219 283]
交叉验证训练集阈值283,最大分数0.862849（较n_estimators=100的样本量学习曲线R^2上升）
前提：Xtrain=354，cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)，那么CVtrian=354*0.8=283；已确定的n_estimators=270。
就是说 当交叉验证训练集样本量达到最大值283时 和 交叉验证训练集阈值283 相同，模型R^2最大0.853370，较n_estimators=100的样本量学习曲线R^2上升。 
这再次证明了：
1、正常来说样本量越大，模型才不容易过拟合，效果越好。
2.1、n_estimators参数控制树的数量决定了模型的学习能力，树的数量越多，模型的学习能力越强；
2.2、树的数量提升对模型的影响有极限，最开始，模型的表现会随着XGB的树的数量一起提升，但到达某个点之后，树的数量越多，模型的效果会逐步下降，这也说明了暴力增加n_estimators不一定有效果。
'''
# In[]:
# 1.3、验证模型效果
time0 = time()
print(XGBR(n_estimators=100,random_state=420, silent=True).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)

print('-'*30)

time0 = time()
print(XGBR(n_estimators=560,random_state=420, silent=True).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)

print('-'*30)

# 可以看到，最后选择的超参数n_estimators=270，R2得分最高0.9216，且耗时也较小0.1147
time0 = time()
print(XGBR(n_estimators=270,random_state=420, silent=True).fit(Xtrain,Ytrain).score(Xtest,Ytest))
print(time()-time0)


# In[]:
# 2、subsample随机抽样的时候抽取的样本比例： （已确定n_estimators=270）
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)
axisx = np.linspace(0,1,20)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=270,subsample=i,random_state=420,silent=True)
    rs.append(CVS(reg,Xtrain,Ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="green",label="XGB")
plt.legend()
plt.show()

# In[]:
# 2.1、细化subsample 方差与泛化误差 学习曲线
axisx = np.linspace(0.05,1,20)
ft.learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, hparam_name="subsample", prev_hparam_value=[270])
# In[]:
# 2.2、再细化subsample 方差与泛化误差 学习曲线
axisx = np.linspace(0.7,1,25)
ft.learning_curve_r2_customize(axisx, Xtrain, Ytrain, cv, hparam_name="subsample", prev_hparam_value=[270])
'''
0.75 0.8762683263611173 0.0004238536122880797
0.75 0.8762683263611173 0.0004238536122880797
0.75 0.8762683263611173 0.0004238536122880797 0.01573338067376706
subsample=0.75时，R^2最高0.8762683263611173，R^2方差最小0.0004238536122880797，泛化误差可控部分0.01573338067376706，那么 CVtrian = 354*0.8=283 * subsample = 212个样本
我们的模型现在正处于样本量过少并且过拟合的状态，根据学习曲线展现出来的规律，我们的训练样本量在200左右的时候，模型的效果有可能反而比更多训练数据的时候好，但这不代表模型的泛化能力在更小的训练样本量下会更强。
正常来说样本量越大，模型才不容易过拟合，现在展现出来的效果，是由于我们的样本量太小造成的一个巧合。从这个角度来看，我们的subsample参数对模型的影响应该会非常不稳定，大概率应该是无法提升模型的泛化能力的，但也不乏提升模型的可能性。
'''
# In[]:
# 2.3、再次以选出的最优参数 运行 样本量学习曲线
ft.plot_learning_curve(XGBR(n_estimators=270,subsample=0.75, random_state=420, silent=True)
                    ,"XGB",Xtrain,Ytrain,ax=None,cv=cv)
plt.show()
'''
样本量阈值[ 28  91 155 219 283]
交叉验证训练集阈值283,最大分数0.876268（是较之前 样本量学习曲线 中R^2最高的）
前提：Xtrain=354，cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)，那么CVtrian=354*0.8=283；已确定的n_estimators=270，subsample=0.75。
就是说 当交叉验证训练集样本量达到最大值283时 和 交叉验证训练集阈值283 相同（实际交叉验证训练集样本量：CVtrian*subsample = 212），模型R^2最大0.876268，是较之前 样本量学习曲线 中R^2最高的。
这再次证明了：1、正常来说样本量越大，模型才不容易过拟合，效果越好。
2.1、n_estimators参数控制树的数量决定了模型的学习能力，树的数量越多，模型的学习能力越强，
2.2、树的数量提升对模型的影响有极限，最开始，模型的表现会随着XGB的树的数量一起提升，但到达某个点之后，树的数量越多，模型的效果会逐步下降，这也说明了暴力增加n_estimators不一定有效果。
2.3、subsample参数在小样本量情况下对模型的影响大概率应该是无法提升模型的泛化能力的，但也不乏提升模型的可能性。
'''
# In[]:
# 2.4、验证模型效果
reg = XGBR(n_estimators=270
           ,subsample=1
           ,random_state=420
           ,silent=True).fit(Xtrain,Ytrain)

print(reg.score(Xtest,Ytest)) # 0.9216018521637588
print(MSE(Ytest,reg.predict(Xtest))) # 7.295252236224126
'''
虽然 超参数n_estimators=270和subsample=0.75都使模型R^2提升了，但在同一个测试集上R^2的提升，并不一定表示 模型的泛化能力 提升。
且由于我们的数据集过少，在小样本量情况下降低抽样的比例subsample超参数反而让数据的效果更低（正常来说），正因如此，保持默认subsample=1。
'''



# In[]:
# 另： 首先我们先来定义一个评分函数，这个评分函数能够帮助我们直接打印Xtrain上的交叉验证结果
def regassess(reg,Xtrain,Ytrain,cv,scoring = ["r2"],show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i] #模型评估指标的名字
                                     ,CVS(reg
                                          ,Xtrain,Ytrain
                                          ,cv=cv,scoring=scoring[i]).mean()))
        score.append(CVS(reg,Xtrain,Ytrain,cv=cv,scoring=scoring[i]).mean())
    return score

# In[]:
reg = XGBR(n_estimators=270,random_state=420,silent=True)
regassess(reg,Xtrain,Ytrain,cv,scoring = ["r2","neg_mean_squared_error"])


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
# In[]:
for i in [0,0.2,0.5,1]:
    time0=time()
    reg = XGBR(n_estimators=270,random_state=420,learning_rate=i,silent=True)
    print("learning_rate = {}".format(i))
    regassess(reg,Xtrain,Ytrain,cv,scoring=["r2","neg_mean_squared_error"])
    print(time()-time0)
    print("\t")
# In[]:
axisx = np.arange(0.05,1,0.05)
rs = []
te = []
for i in axisx:
    reg = XGBR(n_estimators=270,random_state=420,learning_rate=i,silent=True)
    score = regassess(reg,Xtrain,Ytrain,cv,scoring = ["r2","neg_mean_squared_error"],show=True)
    test = reg.fit(Xtrain,Ytrain).score(Xtest,Ytest)
    rs.append(score[0])
    te.append(test)
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,te,c="gray",label="test")
plt.plot(axisx,rs,c="green",label="train")
plt.legend()
plt.show()

# In[]:
# 3.1、由于 eta迭代次数 和 n_estimators 超参数密切相关，需要一起搜索，所以使用GridSearchCV。
#"""
# 运行时间2分多钟
from sklearn.model_selection import GridSearchCV

cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

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
#"""
# In[]:
# 3.2、细化eta 方差与泛化误差 学习曲线：
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

'''
XGB中与梯度提升树的过程相关的四个参数：n_estimators，learning_rate ，silent，subsample。这四个参数的主要目的，其实并不是提升模型表现，更多是了解梯度提升树的原理。
现在来看，我们的梯度提升树可是说是由三个重要的部分组成：
1. 一个能够衡量集成算法效果的，能够被最优化的损失函数Obj。
2. 一个能够实现预测的弱评估器fk(x)
3. 一种能够让弱评估器集成的手段，包括我们讲解的：迭代方法，抽样手段，样本加权等等过程

XGBoost是在梯度提升树的这三个核心要素上运行，它重新定义了损失函数和弱评估器，并且对提升算法的集成手段进行了改进，实现了运算速度和模型效果的高度平衡。
并且，XGBoost将原本的梯度提升树拓展开来，让XGBoost不再是单纯的树的集成模型，也不只是单单的回归模型。
只要我们调节参数，我们可以选择任何我们希望集成的算法，以及任何我们希望实现的功能。
'''




