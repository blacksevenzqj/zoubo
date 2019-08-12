import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1]) # X[y==0,0]取随机数：X[y==0]得到一行，再取第0个元素
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_text, y_train, y_test = train_test_split(X, y, random_state=42)
print(pd.value_counts(y_train, sort=True))

# 使用 BaggingClassifier 包装 DecisionTreeClassifier决策树，就是随机森林

# 使用Bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

'''只针对样本进行随机采样'''
# n_estimators：生成500个决策树子模型；max_samples：每个子模型使用的样本量；bootstrap：是否放回取样（True放回）
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True)
bagging_clf.fit(X_train, y_train)
print(bagging_clf.score(X_text, y_test))

# 理论上，子模型越多，模型预测结果越精确。
bagging_clf2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=5000, max_samples=100, bootstrap=True)
bagging_clf2.fit(X_train, y_train)
print(bagging_clf2.score(X_text, y_test))
