# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:21:09 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

wine = load_wine()

# In[]:
print(wine.data.shape)
print(wine.target.shape)
print(set(wine.target))
unique_label, counts_label = np.unique(wine.target, return_counts=True)
print(unique_label, counts_label)

# In[]:
wine_pd = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
print(wine_pd.info())
print(wine_pd.describe())

# In[]
print(wine.feature_names)
print(wine.target_names)

# In[]:
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3, random_state=1)
print(Xtrain.shape, Ytrain.shape)
print(Xtest.shape, Ytest.shape)
train_unique_label, train_counts_label = np.unique(Ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(Ytest, return_counts=True)
print(train_unique_label, train_counts_label, np.round(train_counts_label / Ytrain.shape, 2))
print(test_unique_label, test_counts_label, np.round(test_counts_label / Ytest.shape, 2))

# In[]:
# 训练集 和 测试集 类别比例相同
sss = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
for train_index, test_index in sss.split(wine.data, wine.target):
#    print("Train:", train_index, "Test:", test_index)
    s_Xtrain, s_Xtest = wine.data[train_index], wine.data[test_index]
    s_ytrain, s_ytest = wine.target[train_index], wine.target[test_index]
    break

print(s_Xtrain.shape, s_ytrain.shape)
print(s_Xtest.shape, s_ytest.shape)
s_train_unique_label, s_train_counts_label = np.unique(s_ytrain, return_counts=True)
s_test_unique_label, s_test_counts_label = np.unique(s_ytest, return_counts=True)
print(s_train_unique_label, s_train_counts_label, np.round(s_train_counts_label / s_ytrain.shape, 2))
print(s_test_unique_label, s_test_counts_label, np.round(s_test_counts_label / s_ytest.shape, 2))

# In[]:
# 一、分类树
# 只设置 random_state
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=23)
# 用 类别比例不同的 训练集 训练模型
clf.fit(Xtrain, Ytrain)
 
score1 = clf.score(Xtest, Ytest) # 预测 类别比例不同的 测试集
print(score1)
score2 = clf.score(s_Xtest, s_ytest) # 预测 类别比例相同的 测试集
print(score2)

# In[]:
s_clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=23)
# 用 类别比例相同的 训练集 训练模型
s_clf.fit(s_Xtrain, s_ytrain)

score1 = s_clf.score(Xtest, Ytest) # 预测 类别比例不同的 测试集
print(score1)

score2 = s_clf.score(s_Xtest, s_ytest) # 预测 类别比例相同的 测试集
print(score2)

# In[]:
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']

import graphviz
import pydotplus
from IPython.display import Image

dot_data = tree.export_graphviz(clf
                                ,feature_names= feature_name
                                ,class_names=["琴酒","雪莉","贝尔摩德"]
                                ,filled=True # 填充颜色
                                ,rounded=True # 节点框形状
                                )
graph = graphviz.Source(dot_data)
graph


# In[]:
print(clf.feature_importances_)
print(clf.feature_importances_.shape)
aaa = clf.feature_importances_.reshape(-1,13)
print(aaa.shape)


temp_import = [*zip(feature_name,clf.feature_importances_)]
print(temp_import)

temp_pd = pd.DataFrame(clf.feature_importances_.reshape(-1,13), columns=feature_name)
print(temp_pd.iloc[0].sort_values(ascending=False))

temp_pd2 = pd.DataFrame(clf.feature_importances_.reshape(-1,1), index=feature_name, columns=['weight'])
temp_pd2.reset_index(drop=False, inplace=True)
temp_pd2.rename(columns={'index':'feature'},inplace=True) 
print(temp_pd2.iloc[temp_pd2['weight'].sort_values(ascending=False).index])


# In[]:
# 同时设置 random_state、splitter
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=23, splitter="random")
# 用 类别比例不同的 训练集 训练模型
clf.fit(Xtrain, Ytrain)
 
score1 = clf.score(Xtest, Ytest) # 预测 类别比例不同的 测试集
print(score1)
score2 = clf.score(s_Xtest, s_ytest) # 预测 类别比例相同的 测试集
print(score2)

# In[]:
s_clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=23, splitter="random")
# 用 类别比例相同的 训练集 训练模型
s_clf.fit(s_Xtrain, s_ytrain)

score1 = s_clf.score(Xtest, Ytest) # 预测 类别比例不同的 测试集
print(score1)

score2 = s_clf.score(s_Xtest, s_ytest) # 预测 类别比例相同的 测试集
print(score2)

# In[]:
dot_data = tree.export_graphviz(clf
                                ,feature_names= feature_name
                                ,class_names=["琴酒","雪莉","贝尔摩德"]
                                ,filled=True # 填充颜色
                                ,rounded=True # 节点框形状
                                )
graph = graphviz.Source(dot_data)
graph



# In[]:
# 剪枝
clf = tree.DecisionTreeClassifier(criterion="entropy", 
                                  random_state=23, 
                                  splitter="random",
                                  max_depth=3,
                                  min_samples_leaf=10,
                                  min_samples_split=10
                                 )
# 用 类别比例不同的 训练集 训练模型
clf.fit(Xtrain, Ytrain)
 
score1 = clf.score(Xtest, Ytest) # 预测 类别比例不同的 测试集
print(score1)
score2 = clf.score(s_Xtest, s_ytest) # 预测 类别比例相同的 测试集
print(score2)

# In[]:
s_clf = tree.DecisionTreeClassifier(criterion="entropy", 
                                    random_state=23, 
                                    splitter="random",
                                    max_depth=3,
                                    min_samples_leaf=10,
                                    min_samples_split=10
                                   )
# 用 类别比例相同的 训练集 训练模型
s_clf.fit(s_Xtrain, s_ytrain)

score1 = s_clf.score(Xtest, Ytest) # 预测 类别比例不同的 测试集
print(score1)

score2 = s_clf.score(s_Xtest, s_ytest) # 预测 类别比例相同的 测试集
print(score2)

# In[]:
dot_data = tree.export_graphviz(clf
                                ,feature_names= feature_name
                                ,class_names=["琴酒","雪莉","贝尔摩德"]
                                ,filled=True # 填充颜色
                                ,rounded=True # 节点框形状
                                )
graph = graphviz.Source(dot_data)
graph


# In[]:
# 学习曲线
import matplotlib.pyplot as plt
train_scores = []
test_scores = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1
                ,criterion="entropy"
                ,random_state=30
                ,splitter="random"
                )
    clf.fit(Xtrain, Ytrain)
    train_scores.append(clf.score(Xtrain,Ytrain))
    test_score = clf.score(Xtest, Ytest)
    test_scores.append(test_score)

plt.plot(range(1,11), train_scores, color="red",label="train")
plt.plot(range(1,11), test_scores, color="blue",label="test")
plt.xticks(range(1,11))
plt.legend()
plt.show()

# In[]:
# 叶子节点索引
print(clf.apply(Xtest))

# In[]:
# min_impurity_decrease 信息增益差值，消耗大。
gini_thresholds = np.linspace(0,0.5,20)

parameters = {'splitter':('best','random')
              ,'criterion':("gini","entropy")
              ,"max_depth":[*range(1,10)]
              ,'min_samples_leaf':[*range(1,50,5)]
              ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
             }

clf = tree.DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(Xtrain,Ytrain)

GS.best_params_
GS.best_score_



# In[]:
# 二、回归树
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
# scoring = "neg_mean_squared_error" 使用负的均方误差评估模型
training_score = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring = "neg_mean_squared_error")
print("Classifiers: ", regressor.__class__.__name__, "Has a training score of", training_score)

kf = KFold(n_splits=10, shuffle=True)
training_score = cross_val_score(regressor, boston.data, boston.target, cv=kf, scoring = "neg_mean_squared_error")
print("Classifiers: ", regressor.__class__.__name__, "Has a training score of", training_score)

# In[]:
# 模型复杂度曲线
maxSampleLeaf = 100
train_scores = []
test_scores = []
for i in range(1, maxSampleLeaf+1):
    dt_reg = DecisionTreeRegressor(min_samples_leaf=i)
    dt_reg.fit(Xtrain,Ytrain)
    y_train_predict = dt_reg.predict(Xtrain)
    train_scores.append(r2_score(Ytrain, y_train_predict)) # 等同于 dt_reg.score(Xtrain, Ytrain)
    test_scores.append(dt_reg.score(Xtest,Ytest))
    
plt.plot([i for i in range(1, maxSampleLeaf+1)], train_scores, label="train")
plt.plot([i for i in range(1, maxSampleLeaf+1)], test_scores, label="test")
plt.xlim(maxSampleLeaf, 1)
plt.legend()
plt.show()

# In[]:
# 基于MSE绘制学习曲线（样本量）
def plot_learning_curve_mse_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])
    
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
    
        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))
        
    plt.plot([i for i in range(1, len(X_train)+1)], 
                               np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train)+1)], 
                               np.sqrt(test_score), label="test")
    plt.legend()
    plt.show()
    
plot_learning_curve_mse_customize(DecisionTreeRegressor(), Xtrain, Xtest, Ytrain, Ytest)

# In[]:
# 基于R^2值绘制学习曲线（样本量）
def plot_learning_curve_r2_customize(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])
    
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(r2_score(y_train[:i], y_train_predict))
    
        y_test_predict = algo.predict(X_test)
        test_score.append(r2_score(y_test, y_test_predict))
        
    plt.plot([i for i in range(1, len(X_train)+1)], 
                               train_score, label="train")
    plt.plot([i for i in range(1, len(X_train)+1)], 
                               test_score, label="test")
    plt.legend()
    plt.axis([0, len(X_train)+1, -0.1, 1.1])
    plt.show()
    
plot_learning_curve_r2_customize(DecisionTreeRegressor(max_depth=3), Xtrain, Xtest, Ytrain, Ytest)

# In[];
# 拟合 正弦函数
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80,1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5) # 5层在图中过拟合
regr_1.fit(X, y)
regr_2.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

































