# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 09:56:24 2018

@author: Administrator
"""

# 导入各种库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

x = np.linspace(0, 1, 100)
print(x)
y = x**x

print(2**2 * 2**3)
print(2**5)
