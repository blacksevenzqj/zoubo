# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:45:20 2019

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.datasets import make_circles, make_moons, make_blobs,make_classification
from sklearn.linear_model import LogisticRegression as LogiR

# In[]:
# 1、演示 支持向量：
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x) 
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    # 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors="k", levels=[-1,0,1], alpha=0.5, linestyles=["--","-","--"]) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_svc_decision_function_new(model, X, Y, score, ax=None):
    if ax is None:
        ax = plt.gca()
        
    # 绘制图像本身分布的散点图
    ax.scatter(X[:, 0], X[:, 1], c=Y
               ,zorder=10
               ,cmap=plt.cm.Paired,edgecolors='k')
    # 绘制支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                facecolors='none', zorder=10, edgecolors='white')
        
    #绘制决策边界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    
    #np.mgrid，合并了我们之前使用的np.linspace和np.meshgrid的用法
    #一次性使用最大值和最小值来生成网格
    #表示为[起始值：结束值：步长]
    #如果步长是复数，则其整数部分就是起始值和结束值之间创建的点的数量，并且结束值被包含在内
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    # np.c_，类似于np.vstack的功能
    # 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
    Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    #填充等高线不同区域的颜色
    ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    #绘制等高线
    ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-1, 0, 1])
    
    #设定坐标轴为不显示
#    ax.set_xticks(())
#    ax.set_yticks(())
    
    #为每张图添加分类的分数   
    ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0')
            , size=15
            , bbox=dict(boxstyle='round', alpha=0.8, facecolor='white')
            	#为分数添加一个白色的格子作为底色
            , transform=ax.transAxes #确定文字所对应的坐标轴，就是ax子图的坐标轴本身
            , horizontalalignment='right' #位于坐标轴的什么方向
           )

# In[]:
dataset = make_classification(n_samples=100,n_features = 2,n_informative=2,n_redundant=0, random_state=5)
X = dataset[0]
y = dataset[1]
kernel_ = 'linear'
# In[]:
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = svm.SVC(kernel=kernel_, C=i, cache_size=5000).fit(X,y)
    score.append(clf.score(X,y))
    
print(max(score), C_range[score.index(max(score))])
linear_c = C_range[score.index(max(score))]
plt.plot(C_range,score)
plt.show()
# In[]:
'''
yi(w∙xi + b) ≥ 1 - ξi
当C趋近于很大时：对误分类的惩罚越大、意味着分类严格不能有错误（泳道越窄）
当C趋近于很小时：对误分类的惩罚越小、意味设可以有更大的错误容忍（泳道越宽）

支持向量的显示 和 C超参数没有关系； 不使用超参数C，任然会显示全部的支持向量（不只是 虚线上的支持向量）
显示全部的支持向量： 反应出模型选择支持向量的过程。
'''
#clf = svm.SVC(kernel=kernel_).fit(X,y) 
clf = svm.SVC(kernel=kernel_, C=linear_c).fit(X,y)
score = clf.score(X,y)

fig, axe = plt.subplots(1,1,figsize=(15,8))
#plot_svc_decision_function(clf, axe)
plot_svc_decision_function_new(clf, X, y, score, axe)
plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show()


# In[]:
n_samples = 100

datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=n_samples, centers=2, random_state=5),
    make_classification(n_samples=n_samples,n_features = 2,n_informative=2,n_redundant=0, random_state=5)
]

Kernel = ["linear"]

# 四个数据集分别是什么样子呢？
for X,Y in datasets:
    plt.figure(figsize=(5,4))
    plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="rainbow")

nrows=len(datasets)
ncols=len(Kernel) + 1

fig, axes = plt.subplots(nrows, ncols,figsize=(10,16))

#第一层循环：在不同的数据集中循环
for ds_cnt, (X,Y) in enumerate(datasets):
    
    #在图像中的第一列，放置原数据的分布
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    
    #第二层循环：在不同的核函数中循环
    #从图像的第二列开始，一个个填充分类结果
    for est_idx, kernel in enumerate(Kernel):
        
        #定义子图位置
        ax = axes[ds_cnt, est_idx + 1]
        
        #建模
        clf = svm.SVC(kernel=kernel, gamma=2).fit(X, Y)
        score = clf.score(X, Y)
        
        plot_svc_decision_function_new(clf, X, Y, score, ax)
        
plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show()



# In[]:
# 2、重要参数class_weight： 样本不均衡处理（SVM中较好的处理方式）
class_1 = 500 #类别1有500个样本，10：1
class_2 = 50 #类别2只有50个
centers = [[0.0, 0.0], [2.0, 2.0]] #设定两个类别的中心
clusters_std = [1.5, 0.5] #设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)

print("y的类别为：%s" % set(y))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow",s=10)
plt.show()
#其中红色点是少数类，紫色点是多数类

# In[]:
# 不设定class_weight
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)
clf_y = clf.predict(X)
clf_decision_scores = clf.decision_function(X)

# 设定class_weight
# class_weight = {1:10} 表示 类别1：10，隐藏了类别0：1这个比例 （不显示设置的类别，权重默认为1）
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)
wclf_y = wclf.predict(X)
wclf_decision_scores = wclf.decision_function(X)

# 给两个模型分别打分看看，这个分数是accuracy准确度
print(clf.score(X,y))
# 做样本均衡之后，我们的准确率下降了，没有样本均衡的准确率更高
print(wclf.score(X,y))

# In[]:
# 首先要有数据分布
plt.figure(figsize=(6,5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow",s=10)

ax = plt.gca() #获取当前的子图，如果不存在，则创建新的子图

# 绘制决策边界的第一步：要有网格
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# 第二步：找出我们的样本点到决策边界的距离
Z_clf = clf.decision_function(xy).reshape(XX.shape)
a = ax.contour(XX, YY, Z_clf, colors='black', levels=[0], alpha=0.5, linestyles=['-']) # 等高线：只画决策边界

Z_wclf = wclf.decision_function(xy).reshape(XX.shape)
b = ax.contour(XX, YY, Z_wclf, colors='red', levels=[0], alpha=0.5, linestyles=['-']) # 等高线：只画决策边界

# 第三步：画图例
'''
a.collections #调用这个等高线对象中画的所有线，返回一个惰性对象

用[*]把它打开试试看
[*a.collections] #返回了一个linecollection对象，其实就是我们等高线里所有的线的列表

现在我们只有一条线，所以我们可以使用索引0来锁定这个对象
a.collections[0]

plt.legend([对象列表],[图例列表],loc)
只要对象列表和图例列表相对应，就可以显示出图例
'''
plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()

# In[]:
# ORC评估指标：
import RocLib as rlb
import matplotlib as mpl
import matplotlib.pyplot as plt

# 精准率： （股票预测） 必须准，可以漏
print("精准率以1：%f" % rlb.precision_score_customize(y, clf_y)) # 不设定class_weight
print("精准率以1class_weight：%f" % rlb.precision_score_customize(y, wclf_y))

print("精准率以0：%f" % rlb.precision_score_customize(y, clf_y, 0)) # 不设定class_weight
print("精准率以0class_weight：%f" % rlb.precision_score_customize(y, wclf_y, 0))


# 召回率： （医疗诊断） 不必准，不能漏
print("召回率以1：%f" % rlb.recall_score_customize(y, clf_y)) # 不设定class_weight
print("召回率以1class_weight：%f" % rlb.recall_score_customize(y, wclf_y))

print("召回率以0：%f" % rlb.recall_score_customize(y, clf_y, 0)) # 不设定class_weight
print("召回率以0class_weight：%f" % rlb.recall_score_customize(y, wclf_y, 0))


# F1分数：
print("F1以1：%f" % rlb.f1_score_customize(y, clf_y)) # 不设定class_weight
print("F1以1class_weight：%f" % rlb.f1_score_customize(y, wclf_y))

print("F1以0：%f" % rlb.f1_score_customize(y, clf_y, 0)) # 不设定class_weight
print("F1以0class_weight：%f" % rlb.f1_score_customize(y, wclf_y, 0))


# 混淆矩阵
print("混淆以1：\n%s" % rlb.confusion_matrix_customize(y, clf_y)) # 不设定class_weight
print("混淆以1class_weight：\n%s" % rlb.confusion_matrix_customize(y, wclf_y))

print("混淆以0：\n%s" % rlb.confusion_matrix_customize(y, clf_y, 0)) # 不设定class_weight
print("混淆以0class_weight：\n%s" % rlb.confusion_matrix_customize(y, wclf_y, 0))

# In[]:
fig, axe = plt.subplots(2,2,figsize=(30,20))
rlb.ComprehensiveIndicatorFigure(y, clf_decision_scores, axe[0], 1)
rlb.ComprehensiveIndicatorSkLibFigure(y, clf_decision_scores, axe[1])



# In[]:
# 概率(probability)与阈值(threshold)
# 测试 predict_proba 
class_1_ = 7
class_2_ = 4
centers_ = [[0.0, 0.0], [1,1]]
clusters_std = [0.5, 1]
X_, y_ = make_blobs(n_samples=[class_1_, class_2_],
                  centers=centers_,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap="rainbow",s=30)
plt.show()

clf_lo = LogiR().fit(X_,y_)
predic = clf_lo.predict(X_)
prob = clf_lo.predict_proba(X_)

prob = pd.DataFrame(prob)

#手动调节阈值，来改变我们的模型效果
for i in range(prob.shape[0]):
    if prob.loc[i,1] > 0.5:
        prob.loc[i,"pred"] = 1
    else:
        prob.loc[i,"pred"] = 0
        
prob["y_true"] = y_
prob = prob.sort_values(by=1,ascending=False)

# In[]:
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R
'''
  0 1
0
1
'''
print(CM(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[0,1])) # 概率0.5的阈值， decision_function为0个阈值。
print(CM(y_, predic))
print(rlb.confusion_matrix_customize(y_, predic))


# In[]:
# 她写的，自定义proba阈值 利用混淆矩阵 的ROC曲线（嘲笑下）， 使用predict_proba阈值 绘制ORC曲线，只能作为理解， 实际还是使用decision_function。
# 如果我们的确需要置信度分数，但不一定非要是概率形式的话，那建议可以将probability设置为False，使用decision_function这个接口而不是predict_proba
clf_proba = svm.SVC(kernel="linear",C=1.0,probability=True).fit(X,y)
probrange = np.linspace(clf_proba.predict_proba(X)[:,1].min(),clf_proba.predict_proba(X)[:,1].max(),num=50,endpoint=False)

from sklearn.metrics import confusion_matrix as CM, recall_score as R
import matplotlib.pyplot as plot

recall = []
FPR = []

for i in probrange:
    y_predict = []
    for j in range(X.shape[0]):
        if clf_proba.predict_proba(X)[j,1] > i:
            y_predict.append(1)
        else:
            y_predict.append(0)
    cm = CM(y,y_predict,labels=[1,0])
    recall.append(cm[0,0]/cm[0,:].sum())
    FPR.append(cm[1,0]/cm[1,:].sum())

recall.sort()
FPR.sort()

plt.plot(FPR,recall,c="red")
plt.plot(probrange+0.05,probrange+0.05,c="black",linestyle="--")
plt.show()


# In[]:
# 实际还是使用decision_function
from sklearn.metrics import roc_curve
FPR, recall, thresholds = roc_curve(y,clf_proba.decision_function(X), pos_label=1)
maxindex = (recall - FPR).tolist().index(max(recall - FPR))
print(thresholds.tolist()[maxindex], FPR.tolist()[maxindex], recall.tolist()[maxindex])

# In[]:
from sklearn.metrics import roc_auc_score as AUC
area = AUC(y,clf_proba.decision_function(X))

# In[]:
#把上述代码放入这段代码中：
plt.figure()
plt.plot(FPR, recall, color='red',
         label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.scatter(FPR[maxindex],recall[maxindex],c="black",s=30)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# In[]:
clf_proba.dual_coef_

# In[]:
clf_proba.support_vectors_

