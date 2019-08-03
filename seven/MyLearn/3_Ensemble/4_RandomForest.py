import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1]) # X[y==0,0]取随机数：X[y==0]得到一行，再取第0个元素
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from sklearn.ensemble import RandomForestClassifier

# RandomForestClassifier 其实就是集成了BaggingClassifier(DecisionTreeClassifier())。
# 即包含决策树的参数、也包含了Bagging的参数。

'''
n_estimators：生成500个决策树子模型；
oob_score：不使用测试数据集，而使用这部分没有取到的样本做测试/验证：Sklearn中使用 oob_score_ 
n_jobs：并行计算，-1使用所有核
max_leaf_nodes：最大叶子节点数
'''
rf_clf = RandomForestClassifier(n_estimators=500, random_state=666, oob_score=True, n_jobs=-1, max_leaf_nodes=16)
rf_clf.fit(X, y)
print(rf_clf.oob_score_)


print("============================================================================================================")


from sklearn.ensemble import ExtraTreesClassifier

# Extra-Trees：即包含决策树的参数、也包含了Bagging的参数。
et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=666, n_jobs=-1)
et_clf.fit(X, y)
print(et_clf.oob_score_)
