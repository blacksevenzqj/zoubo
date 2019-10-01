# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:07:46 2019

@author: dell
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# In[]:
data = load_breast_cancer()
print(data.data.shape)
print(data.target)

# In[]:
rfc = RandomForestClassifier(n_estimators=100,random_state=90)
# 交叉验证的分类默认scoring='accuracy'
score_pre = cross_val_score(rfc,data.data,data.target,cv=10).mean() 

# In[]:
"""
在这里我们选择学习曲线，可以使用网格搜索吗？可以，但是只有学习曲线，才能看见趋势
我个人的倾向是，要看见n_estimators在什么取值开始变得平稳，是否一直推动模型整体准确率的上升等信息
第一次的学习曲线，可以先用来帮助我们划定范围，我们取每十个数作为一个阶段，来观察n_estimators的变化如何
引起模型整体准确率的变化
"""
# 1、第一步：确定 n_estimators 的大致范围
scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1) # 第二个得到的是 n_estimators
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()
# 最后得到： scorel = 0.9684480598046841； n_estimators = 41
 
# In[]:
# 2.1、在上一步n_estimators=41的情况下，进一步细化学习曲线，将n_estimators的范围细化到35至45。
scorel = []
for i in range(35,45):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),([*range(35,45)][scorel.index(max(scorel))])) # 0.9719568317345088 39
plt.figure(figsize=[20,5])
plt.plot(range(35,45),scorel)
plt.show()
# 最后得到： scorel = 0.9719568317345088； n_estimators = 39
# In[]:
# 2.2、注意： 相同的参数，使用cross_val_score结果 和 使用GridSearchCV结果 有些许出入。 
param_grid={'n_estimators':np.arange(30, 50, 1)}
rfc = RandomForestClassifier(random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print(GS.best_params_)
print(GS.best_score_) # {'n_estimators': 39} 0.9718804920913884

# In[]:
"""
有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用学习曲线，看趋势从曲线跑出的结果中选取一个更小的区间，再跑曲线
param_grid = {'n_estimators':np.arange(0, 200, 10)}
param_grid = {'max_depth':np.arange(1, 20, 1)}
param_grid = {'max_leaf_nodes':np.arange(25,50,1)}

对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，再逐渐缩小范围
有一些参数是可以找到一个范围的，或者说我们知道他们的取值和随着他们的取值，模型的整体准确率会如何变化，这
样的参数我们就可以直接跑网格搜索

param_grid = {'criterion':['gini', 'entropy']}
param_grid = {'min_samples_split':np.arange(2, 2+20, 1)}
param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)}
param_grid = {'max_features':np.arange(5,30,1)} 
"""
# In[]:
# 4、调整 max_depth
param_grid = {'max_depth':np.arange(1, 20, 1)}
#   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
#   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10) # 网格搜索
GS.fit(data.data,data.target)
print(GS.best_params_) # 显示调整出来的最佳参数 
print(GS.best_score_) # 返回调整好的最佳参数对应的准确率  {'max_depth': 11} 0.9718804920913884

# In[]:
# 5、调整max_features： 总共30个特征，默认是特征总数开平方≈5，所以从5开始至30
param_grid = {'max_features':np.arange(5,30,1)} 
"""
max_features是唯一一个即能够将模型往左（低方差高偏差）推，也能够将模型往右（高方差低偏差）推的参数。我
们需要根据调参前，模型所在的位置（在泛化误差最低点的左边还是右边）来决定我们要将max_features往哪边调。
现在模型位于图像左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，可用的特征
越多，模型才会越复杂。max_features的默认最小值是sqrt(n_features)，因此我们使用这个值作为调参范围的
最小值。
"""
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print(GS.best_params_)
print(GS.best_score_) # {'max_features': 5} 0.9718804920913884
# 网格搜索返回了max_features的最小值，可见max_features升高之后，模型的准确率降低了。



# In[]：
# 6、调整min_samples_leaf
param_grid={'min_samples_leaf':np.arange(1, 1+10, 1)}
#对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20
#面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围
#如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print(GS.best_params_)
print(GS.best_score_) # {'min_samples_leaf': 1} 0.9718804920913884
# 网格搜索返回了min_samples_leaf的最小值，并且模型整体的准确率还降低了，这和max_depth的情况一致，
# 参数把模型向左推，但是模型的泛化误差上升了。在这种情况下，我们显然是不要把这个参数设置起来的，就让它默认就好了

# In[]:
# 7、调整min_samples_split
param_grid={'min_samples_split':np.arange(2, 2+20, 1)}
 
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print(GS.best_params_)
print(GS.best_score_) # {'min_samples_split': 2} 0.9718804920913884
# 和min_samples_leaf一样的结果，返回最小值并且模型整体的准确率降低了。

# In[]:
# 8、调整Criterion
param_grid = {'criterion':['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print(GS.best_params_)
print(GS.best_score_) # {'criterion': 'gini'} 0.9718804920913884
# 这个使用gini的网格搜索结果分数 和 使用gini的交叉验证的结果分数有些出入，但是相同的意思。

# In[]:
# 9、最终模型：
rfc = RandomForestClassifier(n_estimators=39,random_state=90,criterion='gini')
score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
print(score)
print(score - score_pre)

# In[]:








