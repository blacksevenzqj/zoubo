# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:49:37 2019

@author: dell
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
from sklearn.datasets import make_blobs #自创数据集
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import confusion_matrix as cm, accuracy_score as accuracy, precision_score, f1_score, recall_score as recall, roc_auc_score as auc
import FeatureTools as ft

# In[]:
class_1 = 500 #类别1有500个样本
class_2 = 50 #类别2只有50个
centers = [[0.0, 0.0], [2.0, 2.0]] #设定两个类别的中心
clusters_std = [1.5, 0.5] #设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)

# In[]:
X = pd.DataFrame(X, columns=["X1", "X2"])
y = pd.DataFrame(y, columns=["y"])
data = pd.concat([X, y], axis=1)

# In[]:
ft.Sample_imbalance(data, "y")

# In[]:
Xtrain, Xtest, Ytrain, Ytest = TTS(X,y,test_size=0.3,random_state=420)
# In[]:
ft.sample_category(Ytest, Ytrain)

# In[]:
# 在sklearn下建模#
clf = XGBC().fit(Xtrain,Ytrain)
ypred = clf.predict(Xtest)
ypred_proba = clf.predict_proba(Xtest) 
# In[]:
print(clf.score(Xtest,Ytest)) # 默认模型评估指标 - 准确率
print(cm(Ytest,ypred,labels=[1,0])) # 少数类写在前面
print(recall(Ytest,ypred))
print(auc(Ytest,clf.predict_proba(Xtest)[:,1]))

# In[]:
clf = XGBC(scale_pos_weight=10).fit(Xtrain,Ytrain) # 负:0/正:1 样本比例
ypred = clf.predict(Xtest)
ypred_proba = clf.predict_proba(Xtest) 
# In[]:
print(clf.score(Xtest,Ytest)) #默认模型评估指标 - 准确率
print(cm(Ytest,ypred,labels=[1,0])) # 少数类写在前面
print(recall(Ytest,ypred))
print(auc(Ytest,clf.predict_proba(Xtest)[:,1]))

# In[]:
# 随着样本权重逐渐增加，模型的recall,auc和准确率如何变化？
for i in [1,5,10,20,30]:
    clf_ = XGBC(scale_pos_weight=i).fit(Xtrain,Ytrain)
    ypred_ = clf_.predict(Xtest)
    print(i)
    print("\tAccuracy:{}".format(clf_.score(Xtest,Ytest)))
    print("\tRecall:{}".format(recall(Ytest,ypred_)))
    print("\tAUC:{}".format(auc(Ytest,clf_.predict_proba(Xtest)[:,1])))
    
# In[]:
# 负/正样本比例
clf_ = XGBC(scale_pos_weight=20).fit(Xtrain,Ytrain)
ypred_ = clf_.predict(Xtest)
ypred_proba_ = clf.predict_proba(Xtest) 
# In[]:
print(clf_.score(Xtest,Ytest)) #默认模型评估指标 - 准确率
print(cm(Ytest,ypred_,labels=[1,0])) # 少数类写在前面
print(recall(Ytest,ypred_))
print(auc(Ytest,clf_.predict_proba(Xtest)[:,1]))



# In[]:
# xgboost原生库：
dtrain = xgb.DMatrix(Xtrain,Ytrain)
dtest = xgb.DMatrix(Xtest,Ytest)
# In[]:
# 看看xgboost库自带的predict接口
param = {'silent':True,'objective':'binary:logistic',"eta":0.1,"scale_pos_weight":1}
num_round = 100
bst = xgb.train(param, dtrain, num_round)
# In[]:
preds = bst.predict(dtest) # 直接返回概率（少数类别1的概率）
# In[]:
# 自己设定阈值
ypred = preds.copy()
# In[]
ypred[ypred >= 0.5] = 1
ypred[ypred < 0.5] = 0


# In[]:
# 写明参数
scale_pos_weight = [1,5,10,20]
names = ["negative vs positive: 1"
         ,"negative vs positive: 5"
         ,"negative vs positive: 10"
         ,"negative vs positive: 20"
         ]

# In[]:
[*zip(names,scale_pos_weight)]
# In[]:
for name,i in zip(names,scale_pos_weight):
    param = {'silent':True,'objective':'binary:logistic'
            ,"eta":0.1,"scale_pos_weight":i}
    num_round = 100
    clf = xgb.train(param, dtrain, num_round)
    preds = clf.predict(dtest)
    ypred = preds.copy()
    ypred[preds > 0.5] = 1
    ypred[ypred != 1] = 0
    print(name)
    print("\tAccuracy:{}".format(accuracy(Ytest,ypred)))
    print("\tRecall:{}".format(recall(Ytest,ypred)))
    print("\tAUC:{}".format(auc(Ytest,preds)))

# In[]:
param = {'silent':True,'objective':'binary:logistic'
            ,"eta":0.1,"scale_pos_weight":10}
num_round = 100
clf = xgb.train(param, dtrain, num_round)
preds = clf.predict(dtest)
ypred = preds.copy()
ypred[preds > 0.5] = 1
ypred[ypred != 1] = 0
print(name)
print("\tAccuracy:{}".format(accuracy(Ytest,ypred)))
print("\tRecall:{}".format(recall(Ytest,ypred)))
print("\tAUC:{}".format(auc(Ytest,preds)))
# In[]:
# 本质上来说，scale_pos_weight参数是通过调节预测的概率值来调节，大家可以通过查看bst.predict(Xtest)返回的结果来观察概率受到了怎样的影响。
# 所以根据 scale_pos_weight参数 结果 再调整 阈值 的意义不大，直接就调整 scale_pos_weight参数 即可。
thresholds = np.arange(ypred.min(),ypred.max(),0.01)
    
precisions = []
recalls = []
f1Scores = []

for threshold in thresholds:
    my_predict = np.array(ypred >= threshold, dtype='int')
    precisions.append(precision_score(Ytest, my_predict))
    recalls.append(recall(Ytest, my_predict))
    f1Scores.append(f1_score(Ytest, my_predict))

#print("阈值：", thresholds)
#print("准确率：", precisions)
#print("召回率：", recalls)
#print("F1分数：", f1Scores)


# matplotlib 图表中文显示
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize = (12,4))
# 1、A图
ax1 = fig.add_subplot(1,2,1)
plt.plot(thresholds, precisions, color = 'blue', label='精准率')
plt.plot(thresholds, recalls, color='black', label='召回率')
plt.plot(thresholds, f1Scores, color='green', label='F1分数')
plt.legend()  # 图例
plt.xlabel('阈值')  # x轴标签
plt.ylabel('精准率、召回率、F1分数') # y轴标签

# 2、B图
ax2 = fig.add_subplot(1,2,2)
plt.plot(precisions, recalls, color='purple', label='P-R曲线')
plt.legend()  # 图例
plt.xlabel('精准率')  # x轴标签
plt.ylabel('召回率') # y轴标签

plt.show()


    