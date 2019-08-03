import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1]) # X[y==0,0]取随机数：X[y==0]得到一行，再取第0个元素
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

# 使用 BaggingClassifier 包装 DecisionTreeClassifier决策树，就是随机森林

'''
n_estimators：生成500个决策树子模型；
max_samples：每个子模型使用的样本量；
bootstrap：是否放回取样（True放回）
oob_score：不使用测试数据集，而使用这部分没有取到的样本做测试/验证：Sklearn中使用 oob_score_ 
n_jobs：并行计算，-1使用所有核
max_features：随机取样的特征个数
bootstrap_features：是否放回特征取样（True放回）
'''

# 使用oob
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# 1、只针对样本进行随机采样
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, oob_score=True)
bagging_clf.fit(X, y) # 将所有数据都放入进行训练
print(bagging_clf.oob_score_)

print("------------------------------------------------------------------------------------------------------------")

# n_jobs：并行计算，-1使用所有核
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True,
                                n_jobs=-1)
bagging_clf.fit(X, y) # 将所有数据都放入进行训练
print(bagging_clf.oob_score_)


print("============================================================================================================")


# 2、针对特征进行采样
# max_features：随机取样的特征个数，由于本例中只有2个特征，所有这里=1
# bootstrap_features：是否放回特征取样（True放回）

# 将max_samples设置为总样本量500，意思就是没有进行 样本的随机放回采样
random_subspaces_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=500,
                                bootstrap=True, oob_score=True,
                                n_jobs=-1,
                                max_features=1, bootstrap_features=True)
random_subspaces_clf.fit(X, y) # 将所有数据都放入进行训练
print(random_subspaces_clf.oob_score_)


print("============================================================================================================")


# 3、即针对样本，又针对特征进行随机采样
# 将max_samples设置为总样本量100，进行样本的随机采样
# 设置max_features 和 bootstrap_features 进行特征的随机采样
random_patches_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500, max_samples=100,
                                bootstrap=True, oob_score=True,
                                n_jobs=-1,
                                max_features=1, bootstrap_features=True)
random_patches_clf.fit(X, y) # 将所有数据都放入进行训练
print(random_patches_clf.oob_score_)


