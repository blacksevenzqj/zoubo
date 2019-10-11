# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:15:02 2019

@author: dell
"""

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # 精确性分数
 
# In[]:
data = load_breast_cancer()#乳腺癌数据集
X = data.data
y = data.target
 
X.data.shape # (569, 30)
 
lrl1 = LR(penalty="l1",solver="liblinear",C=0.5,max_iter=1000)
 
lrl2 = LR(penalty="l2",solver="liblinear",C=0.5,max_iter=1000)
 
# coef_： 每个特征 所对应的 系数
lrl1 = lrl1.fit(X,y)
print(lrl1.coef_)
print((lrl1.coef_ != 0).sum(axis=1)) # array([10]) 30个特征中有10个特征的系数不为0
 
lrl2 = lrl2.fit(X,y)
#print(lrl2.coef_)

# In[]:
# L1、L2 学习曲线：
l1 = []
l2 = []
l1test = []
l2test = []
 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
 
for i in np.linspace(0.05,1.5,19):
    lrl1 = LR(penalty="l1",solver="liblinear",C=i,max_iter=1000)
    lrl2 = LR(penalty="l2",solver="liblinear",C=i,max_iter=1000)
    
    lrl1 = lrl1.fit(Xtrain,Ytrain)
    l1.append(accuracy_score(lrl1.predict(Xtrain),Ytrain))
    l1test.append(accuracy_score(lrl1.predict(Xtest),Ytest))
    
    lrl2 = lrl2.fit(Xtrain,Ytrain)
    l2.append(accuracy_score(lrl2.predict(Xtrain),Ytrain))
    l2test.append(accuracy_score(lrl2.predict(Xtest),Ytest))
 
graph = [l1,l2,l1test,l2test]
color = ["green","black","lightgreen","gray"]
label = ["L1","L2","L1test","L2test"]    
 
plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,1.5,19),graph[i],color[i],label=label[i])
plt.legend(loc=4) # 图例的位置在哪里?4表示，右下角
plt.show()



# In[]:
# 高效的嵌入法embedded： L1正则化 的 特征选择 的学习曲线：
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

# In[]:
data = load_breast_cancer()
data.data.shape
 
LR_ = LR(solver="liblinear",C=0.9,random_state=420)
print(cross_val_score(LR_,data.data,data.target,cv=10).mean())
 
sfmf = SelectFromModel(LR_,norm_order=1).fit(data.data,data.target)
X_embedded_index = sfmf.get_support(indices=True) # 降维后特征的 原索引
X_embedded = sfmf.transform(data.data)
print(X_embedded.shape) # (569, 9)
print(cross_val_score(LR_,X_embedded,data.target,cv=10).mean()) # 0.9368323826808401

# In[]:
# 1、调整参数threshold（无效方法）
'''
调节SelectFromModel这个类中的参数threshold，这是嵌入法的阈值，表示删除所有参数的绝对值低于这个阈
值的特征。现在threshold默认为None，所以SelectFromModel只根据L1正则化的结果来选择了特征，即选择了所
有L1正则化后参数不为0的特征。我们此时，只要调整threshold的值（画出threshold的学习曲线），就可以观察
不同的threshold下模型的效果如何变化。一旦调整threshold，就不是在使用L1正则化选择特征，而是使用模型的
属性.coef_中生成的各个特征的系数来选择。coef_虽然返回的是特征的系数，但是系数的大小和决策树中的
feature_ importances_以及降维算法中的可解释性方差explained_vairance_概念相似，其实都是衡量特征的重要
程度和贡献度的，因此SelectFromModel中的参数threshold可以设置为coef_的阈值，即可以剔除系数小于
threshold中输入的数字的所有特征
'''
fullx = []
fsx = []
 
threshold = np.linspace(0,abs((LR_.fit(data.data,data.target).coef_)).max(),20)
 
k=0
for i in threshold:
    X_embedded = SelectFromModel(LR_,threshold=i).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=5).mean())
    fsx.append(cross_val_score(LR_,X_embedded,data.target,cv=5).mean())
    print((threshold[k],X_embedded.shape[1]))
    k+=1
    
plt.figure(figsize=(20,5))
plt.plot(threshold,fullx,label="full")
plt.plot(threshold,fsx,label="feature selection")
plt.xticks(threshold)
plt.legend()
plt.show()
'''
然而，这种方法其实是比较无效的，大家可以用学习曲线来跑一跑：当threshold越来越大，被删除的特征越来越
多，模型的效果也越来越差，模型效果最好的情况下需要保证有17个以上的特征。实际上我画了细化的学习曲线，
如果要保证模型的效果比降维前更好，我们需要保留25个特征，这对于现实情况来说，是一种无效的降维：需要
30个指标来判断病情，和需要25个指标来判断病情，对医生来说区别不大。
'''

# In[];
# 2、调整逻辑回归的类LR_的L1取值： 通过画C的学习曲线来实现
fullx = []
fsx = []
X_embedded_indexs = []
C = np.arange(0.01, 10.01, 0.5)
 
for i in C:
    LR_ = LR(solver="liblinear",C=i,random_state=420)
    
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    
    
    sfmf = SelectFromModel(LR_,norm_order=1).fit(data.data,data.target)
    X_embedded_index = sfmf.get_support(indices=True) # 特征选择后 特征的 原列位置索引
    X_embedded = sfmf.transform(data.data)
    X_embedded_indexs.append(X_embedded_index)
    fsx.append(cross_val_score(LR_,X_embedded,data.target,cv=10).mean())
    
print(max(fsx),C[fsx.index(max(fsx))]) # 0.9563164376458386 7.01
print(X_embedded_indexs[fsx.index(max(fsx))]) # [ 0  5  6 11 20 25 26 27 28]
 
plt.figure(figsize=(20,5))
plt.plot(C,fullx,label="full")
plt.plot(C,fsx,label="feature selection")
plt.xticks(C)
plt.legend()
plt.show()

# In[]:
# 继续细化学习曲线（根据上一步得到的 C参数的最优值7.01 进一步细化）
fullx = []
fsx = []
 
C = np.arange(6.05,7.05,0.005)
 
for i in C:
    LR_ = LR(solver="liblinear",C=i,random_state=420)
    
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    
    X_embedded = SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
    fsx.append(cross_val_score(LR_,X_embedded,data.target,cv=10).mean())
    
print(max(fsx),C[fsx.index(max(fsx))]) # 0.9580405755768732 6.069999999999999
 
plt.figure(figsize=(20,5))
plt.plot(C,fullx,label="full")
plt.plot(C,fsx,label="feature selection")
plt.xticks(C)
plt.legend()
plt.show()
 
#验证模型效果：降维之前
LR_ = LR(solver="liblinear",C=6.069999999999999,random_state=420)
print(cross_val_score(LR_,data.data,data.target,cv=10).mean()) # 0.947360859044162
 
#验证模型效果：降维之后
LR_ = LR(solver="liblinear",C=6.069999999999999,random_state=420)
X_embedded = SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
print(cross_val_score(LR_,X_embedded,data.target,cv=10).mean()) # 0.9580405755768732
 
print(X_embedded.shape) # (569, 10)



# In[]:
# 步长控制参数max_iter： 
l2 = []
l2test = []
 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
 
for i in np.arange(1,201,10):
    lrl2 = LR(penalty="l2",solver="liblinear",C=0.9,max_iter=i)
    lrl2 = lrl2.fit(Xtrain,Ytrain)
    l2.append(accuracy_score(lrl2.predict(Xtrain),Ytrain))
    l2test.append(accuracy_score(lrl2.predict(Xtest),Ytest))
    
graph = [l2,l2test]
color = ["black","gray"]
label = ["L2","L2test"]
    
plt.figure(figsize=(20,5))
for i in range(len(graph)):
    plt.plot(np.arange(1,201,10),graph[i],color[i],label=label[i])
plt.legend(loc=4)
plt.xticks(np.arange(1,201,10))
plt.show()
 
#我们可以使用属性.n_iter_来调用本次求解中真正实现的迭代次数
lr = LR(penalty="l2",solver="liblinear",C=0.9,max_iter=300).fit(Xtrain,Ytrain)
lr.n_iter_ # array([24], dtype=int32)  只迭代了24次就达到收敛



# In[]:
# 重要参数solver & multi_class： 
from sklearn.datasets import load_iris
iris = load_iris()
iris.target # 三分类数据集

# 打印两种multi_class模式下的训练分数
# %的用法，用%来代替打印的字符串中，想由变量替换的部分。%.3f表示，保留三位小数的浮点数。%s表示，字符串。
# 字符串后的%后使用元祖来容纳变量，字符串中有几个%，元祖中就需要有几个变量
for multi_class in ('multinomial', 'ovr'):
    clf = LR(solver='sag', max_iter=100, random_state=42,
                             multi_class=multi_class).fit(iris.data, iris.target)
    print("training score : %.3f (%s)" % (clf.score(iris.data, iris.target), multi_class))

