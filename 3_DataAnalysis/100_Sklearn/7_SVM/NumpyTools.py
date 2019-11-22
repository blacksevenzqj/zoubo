# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:17:19 2019

@author: dell
"""

import numpy as np

# In[]:
# 1、triu_indices_from 返回矩阵的 上三角矩阵 的索引
'''
两个行向量，按列组合 形成 原始矩阵 的 上三角矩阵的索引
[0 0 0 0 1 1 1 2 2 3]
[0 1 2 3 1 2 3 2 3 3]
'''
def upper_triangular_matrixv(data):
    a = np.triu_indices_from(np.zeros_like(data))[0]
    b = np.triu_indices_from(np.zeros_like(data))[1]
    c = np.vstack((a, b)) # 堆叠上三角矩阵索引
    
    result = []
    for i in range(c.shape[1]):
        result.append(data[c[0,i], c[1,i]])
    
    return c, result

# In[]:
# numpy数组的拼接、合并  
# https://blog.csdn.net/qq_39516859/article/details/80666070
# https://blog.csdn.net/qq_38150441/article/details/80488800

# In[]:
# 测试 column_stack 与 hstack 的区别： 1维有区别，2维之后相同。
def column_stack_and_hstack():
    # 一维
    a=[1, 2, 3]
    b=[11, 22, 33]
    print(a)
    print(b)
    
    c = np.column_stack((a,b))
    print(c)
    
    d = np.hstack((a,b))
    print(d)
    
    print(30*"-")
    
    # 一维
    a = np.array([1, 2, 3])
    b = np.array([11, 22, 33])
    print(a)
    print(b)
    
    c = np.column_stack((a,b))
    print(c)
    
    d = np.hstack((a,b))
    print(d)
    
    print(30*"=")
    
    # 二维：
    a=[[1],[2],[3]]
    b=[[11],[12],[13]]
    print(a)
    print(b)
    
    c = np.column_stack((a,b))
    print(c)
    
    d = np.hstack((a,b))
    print(d)
    
    print(30*"-")
    
    # 二维：
    a = np.array([[1],[2],[3]])
    b = np.array([[11],[12],[13]])
    print(a)
    print(b)
    
    c = np.column_stack((a,b))
    print(c)
    
    d = np.hstack((a,b))
    print(d)
    
    print(30*"=")
    
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    b = np.array([[10,11,12],[13,14,15],[16,17,18]])
    print(a)
    print(b)
    
    c = np.column_stack((a,b))
    print(c)
    
    d = np.hstack((a,b))
    print(d)
    
    print(30*"=")
    
    a=[1, 2, 3, 4, 5, 6]
    print(a)
    c = np.column_stack(a)
    print(c)
    d = np.hstack(a)
    print(d)
    
    print(30*"-")

    a = [[1,2,3],[4,5,6],[7,8,9]]
    print(a)
    c = np.column_stack(a)
    print(c)
    d = np.hstack(a)
    print(d)
    
    

