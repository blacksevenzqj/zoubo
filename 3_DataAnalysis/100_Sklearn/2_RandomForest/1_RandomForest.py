# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:50:21 2019

@author: dell
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

wine = load_wine()
 
wine.data
wine.target

# In[]:
#实例化
#训练集带入实例化的模型去进行训练，使用的接口是fit
#使用其他接口将测试集导入我们训练好的模型，去获取我们希望过去的结果（score.Y_test）
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
 
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(Xtrain,Ytrain)
score_c = clf.score(Xtest,Ytest)
rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(Xtrain,Ytrain)
score_r = rfc.score(Xtest,Ytest)
 
print("Single Tree:{}".format(score_c)
      ,"Random Forest:{}".format(score_r)
     )

# In[]:
#目的是带大家复习一下交叉验证
#交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
rfc_l = []
clf_l = []
 
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc,Xtrain,Ytrain,cv=10).mean()
    rfc_l.append(rfc_s)
    
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf,wine.data,wine.target,cv=10).mean()
    clf_l.append(clf_s)
    
plt.plot(range(1,11),rfc_l,label = "Random Forest train")
plt.plot(range(1,11),clf_l,label = "Decision Tree train")
plt.legend()
plt.show()
 
#是否有注意到，单个决策树的波动轨迹和随机森林一致？
#再次验证了我们之前提到的，单个决策树的准确率越高，随机森林的准确率也会越高

# In[]:
#【TIME WARNING: 2mins 30 seconds】#
#superpa = []
#for i in range(200):
#    rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
#    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
#    superpa.append(rfc_s)
#print(max(superpa),superpa.index(max(superpa))+1)#打印出：最高精确度取值，max(superpa))+1指的是森林数目的数量n_estimators
#plt.figure(figsize=[20,5])
#plt.plot(range(1,201),superpa)
#plt.show()

# In[]:
import numpy as np
from scipy.special import comb
 
np.array([comb(25,i)*(0.2**i)*((1-0.2)**(25-i)) for i in range(13,26)]).sum()

# In[]:
#随机森林的重要属性之一：estimators，查看森林中树的状况
rfc = RandomForestClassifier(n_estimators=20,random_state=2)
rfc.fit(Xtrain, Ytrain)
rfc.estimators_[0].random_state
for i in range(len(rfc.estimators_)):
    print(rfc.estimators_[i].random_state)
    
# In[]:
#无需划分训练集和测试集
rfc = RandomForestClassifier(n_estimators=25,oob_score=True)#默认为False
rfc.fit(wine.data,wine.target)
#重要属性oob_score_
#rfc.oob_decision_function_ # 预测值 
rfc.oob_score_ # 准确率

# In[]:
#大家可以分别取尝试一下这些属性和接口
rfc = RandomForestClassifier(n_estimators=25)
rfc.fit(Xtrain, Ytrain)
score = rfc.score(Xtest,Ytest)
print(score)

rfc.feature_importances_#结合zip可以对照特征名字查看特征重要性，参见上节决策树
rfc.apply(Xtest)#apply返回每个测试样本所在的叶子节点的索引
rfc.predict(Xtest)#predict返回每个测试样本的分类/回归结果
rfc.predict_proba(Xtest) # 概率

# In[]:
import numpy as np
x = np.linspace(0,1,20)
y = []
for epsilon in np.linspace(0,1,20):
    E = np.array([comb(25,i)*(epsilon**i)*((1-epsilon)**(25-i)) for i in range(13,26)]).sum()      
    y.append(E)
plt.plot(x,y,"o-",label="when estimators are different")
plt.plot(x,x,"--",color="red",label="if all estimators are same")
plt.xlabel("individual estimator's error")
plt.ylabel("RandomForest's error")
plt.legend()
plt.show()



# In[]:
# 回归树
from sklearn.datasets import load_boston#一个标签是连续西变量的数据集
from sklearn.model_selection import cross_val_score#导入交叉验证模块
from sklearn.ensemble import RandomForestRegressor#导入随机森林回归系

# In[]:
# sklearn当中的模型评估指标（打分）列表
import sklearn
sorted(sklearn.metrics.SCORERS.keys()) # 这些指标是scoring可选择的参数

# In[]:
boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100,random_state=0,oob_score=True) # 实例化
regressor.fit(boston.data, boston.target)
regressor.oob_score_ # R方
#regressor.oob_prediction_ # 预测值

# In[]:
# 如果不写 neg_mean_squared_error，回归评估默认是R平方
regressor = RandomForestRegressor(n_estimators=100,random_state=0) # 实例化
scores = cross_val_score(regressor, Xtrain, Ytrain, cv=10
               ,scoring = "neg_mean_squared_error" # 最小均方差
               )
print(scores)

# In[]:
# 手动交叉验证
from sklearn.model_selection import KFold, StratifiedKFold

for k in range(50, 101, 50):
    for i in range(1, 21):
        train_scores = []
        test_scores = []
        regressor = RandomForestRegressor(n_estimators=k,random_state=0,min_samples_leaf=i,oob_score=True)
        
        sss = KFold(n_splits=10, random_state=None, shuffle=False)
        for train_index, test_index in sss.split(boston.data, boston.target):
        #    print("Train:", train_index, "Test:", test_index)
            undersample_Xtrain, undersample_Xtest = boston.data[train_index], boston.data[test_index]
            undersample_ytrain, undersample_ytest = boston.target[train_index], boston.target[test_index]
            regressor.fit(undersample_Xtrain, undersample_ytrain)
            
            train_scores.append(regressor.oob_score_) # R^2
            test_scores.append(regressor.score(undersample_Xtest, undersample_ytest)) # R^2
        
        plt.plot(range(1,11), train_scores, color="red",label="train")
        plt.plot(range(1,11), test_scores, color="blue",label="test")
        plt.xticks(range(1,11))
        plt.legend()
        plt.show()








