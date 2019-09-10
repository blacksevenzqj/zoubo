import numpy as np


# 1、triu_indices_from 返回矩阵的 上三角矩阵 的索引
data = np.random.rand(4,4)
print(data)
print(np.triu_indices_from(np.zeros_like(data))[0])
print(np.triu_indices_from(np.zeros_like(data))[1])
'''
两个行向量，按列组合 形成 原始矩阵 的 上三角矩阵的索引
[0 0 0 0 1 1 1 2 2 3]
[0 1 2 3 1 2 3 2 3 3]
'''