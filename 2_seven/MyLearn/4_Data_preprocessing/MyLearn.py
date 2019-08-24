from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import numpy as np


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 非线性转换

# 1、分位数变换（百分位秩）：映射到均匀分布
quantile_transformer = QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)

# print(np.sort(X_train[:, 0]))
'''
np.percentile()是对数据进行分位数处理，即X_train[:, 0]先选择了X_train的第一列所有数据，然后np.percentile()选择了
排序后的0%、25%、50%、75%和100%的数据元素，总的来说，这就是一种抽查数据的手段
'''
print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))

print(np.percentile(X_test[:, 0], [0, 25, 50, 75, 100]))
print(np.percentile(X_test_trans[:, 0], [0, 25, 50, 75, 100]))


print("============================================================================================================")


# 2、幂变换（Tukey正态分布打分）：映射到高斯分布
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)

print(X_train_trans)
print(quantile_transformer.quantiles_)