# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pydotplus
from sklearn.metrics import confusion_matrix


print(np.log2(8) + np.log2(8))
print(np.log2(64))
print(np.log10(500) + np.log10(500))
print(np.log10(500 * 500))
print(2 * np.log10(500))
print(np.log10(500)-np.log10(200))
print(np.log10(800)-np.log10(500))

# print(np.log(0))

# feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
# for i, pair in enumerate(feature_pairs):
#     print(i, pair)

# print(np.random.rand(10) * 6)
#
# print(np.linspace(-3, 3, 50))

# f = np.array([123, 456, 789])
# print(np.max(f))
# f -= np.max(f)
# print(f)


# print(np.exp([1,-2,0]))
# print(np.sum(np.exp([1,-2,0])))
# a = np.exp([1,-2,0])
# b = np.sum(np.exp([1,-2,0]))
# print(a / b)

# table = pd.DataFrame(np.zeros((4,2)), index=['a','b','c','d'], columns=['left', 'right'])
# print(table.values)
# print(table)
# table["center"] = [1,1,1,1]
# print(table)
# table.to_csv("C:\\Users\\dell\\Desktop\\abc\\sample.csv")


# a = [1,2,3]
# b = a
# b.append(4)
# print(a)
#
# a = [1,2,3]
# b = a.copy()
# print(id(a), id(b))
# a.append(4)
# print(b)


# e = 2893 * 1929 / 3463
# print(e)


# a = np.logspace(2.0, 3.0, num=4)
# print(a)
#
# b = np.logspace(0, 9, 10, base=2)
# print(b)
#
# print(2**9)
# np.logspace(2.0, 3.0, num=4, endpoint=False)
#
# np.logspace(2.0, 3.0, num=4, base=2.0)

# print(np.log10(0.001))
# print(np.log10(0.01))
# print(np.exp(-1.9))
# print(10**-1.9)
#
# print(np.log10(0.014236350631724466))
# print(np.log10(0.012589254117941675))
# print(np.log10(0.14956861922263506))

# feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
# for i, pair in enumerate(feature_pairs):
#     print(i, pair)


# score_list = np.random.randint(25, 100, size=20)
# print(score_list)
#
# bins = [0, 59, 70, 80, 100]
#
# score_cut = pd.cut(score_list, bins, labels=False)
# print(type(score_cut), score_cut.dtype)  # <class 'numpy.ndarray'> int64
# print(score_cut) # [2 0 1 3 2 3 2 0 3 2 1 0 0 0 2 0 3 0 3 1]
#
# score_cut = pd.cut(score_list, bins)
# print(type(score_cut), score_cut.dtype)  # <class 'pandas.core.arrays.categorical.Categorical'> category
# print(score_cut) # [(0, 59], (0, 59], (0, 59], (59, 70], (59, 70], ...
# print(score_cut.value_counts())
#
# num = pd.Series(np.random.randint(25, 100, size=20))
# # print(num)
# score_cut = pd.cut(num, bins)
# # print(score_cut)
# print(type(score_cut), score_cut.dtype) # <class 'pandas.core.series.Series'> category
#
# score_cut = pd.cut(num, bins, labels=False)
# # print(score_cut)
# print(type(score_cut), score_cut.dtype) # <class 'pandas.core.series.Series'> int64




# a = np.array([[1624,  1269,  2893],
#                [305,   265,   570],
#                [1929,  1534,  3463]])
#
# w00 = (a[0,0] - (a[0,2] * a[2, 0] / a[2,2]))**2 / (a[0,2] * a[2, 0] / a[2,2])
#
# w01 = (a[0,1] - (a[0,2] * a[2, 1] / a[2,2]))**2 / (a[0,2] * a[2, 1] / a[2,2])
#
# w10 = (a[1,0] - (a[1,2] * a[2, 0] / a[2,2]))**2 / (a[1,2] * a[2, 0] / a[2,2])
#
# w11 = (a[1,1] - (a[1,2] * a[2, 1] / a[2,2]))**2 / (a[1,2] * a[2, 1] / a[2,2])
#
# print(w00 + w01 + w10 + w11)



# print(2.91570161e-001)
# print(7.56633888e-009)
# print(2.91570161 * (10 ** -2))
# print(10 ** -2)


# y_true = [2, 1, 0, 1, 2, 0]
# y_pred = [2, 0, 0, 1, 2, 1]
# c = confusion_matrix(y_true, y_pred, labels=[0,1,2])
# print(c)
#
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# c = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
# print(c)

# e = 0.6
# e1 = (1 - e) / e
# print(e1)
# a = np.log2(e1)
# print(a)


e = 0.55
e1 = (1 - e) / e
print(e1)
a = np.log2(e1)
print(a)

# print(np.exp(-0.58))