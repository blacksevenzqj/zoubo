# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit

# 行为样本，列为特征
# 分几折，循环几次

# In[1]:
# 1、K折交叉验证：
# 1.1、KFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([1, 1, 3, 4, 3, 5])
kf = KFold(n_splits=3)  # 分3折，2折训练=(2/3)*6=4样本； 1折测试=(1/3)*6=2样本。 严格按照 n-1折训练集，1折测试划分
print(kf.get_n_splits(X)) # int
print(kf)
for train_index, test_index in kf.split(X):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

# %%
# 1.2、GroupKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([3, 3, 3, 2, 3, 2])
groups = np.array([1, 2, 1, 2, 2, 2])
group_kfold = GroupKFold(n_splits=2)  # 分2折，根据groups进行划分，不根据y（如果划分折数大于分组数，报错）
group_kfold.get_n_splits(X, y, groups)
print(group_kfold)
for train_index, test_index in group_kfold.split(X, y, groups):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

# %%
# 1.3、StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
print((2 / 3) * 7)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
y = np.array([1, 1, 1, 2, 2, 3, 3, 4])
skf = StratifiedKFold(n_splits=3)  # 分3折，以训练集 和 测试集 各类别样本比例划分
skf.get_n_splits(X, y)
print(skf)
for train_index, test_index in skf.split(X, y):
    print("Train Index:", train_index, ",Test Index:", test_index)  # train_index为2折，test_index为1折
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(len(y_train), len(y_test))
    train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
    print(train_unique_label, train_counts_label / len(y_train))
    print((train_counts_label / len(y_train)) * len(y_train))
    test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)
    print(test_unique_label, test_counts_label / len(y_test))
    print((test_counts_label / len(y_test)) * len(y_test))
#    break


# In[2]:
# 2、随机划分法：
# 2.1、ShuffleSplit 把数据集打乱顺序，然后划分测试集和训练集，训练集额和测试集的比例随机选定(训练集和测试集的比例的和可以小于1)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([2, 2, 3, 3, 4, 4])
rs = ShuffleSplit(n_splits=2, test_size=.25, random_state=0)  # 分2折，定义test_size=.25，那么train_size=0.75
print(rs.get_n_splits(X))
print(rs)
for train_index, test_index in rs.split(X, y):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(X_train,X_test,y_train,y_test)

print("===============================================================")

rs = ShuffleSplit(n_splits=3, train_size=.5, test_size=.25, random_state=0) # 训练集和测试集的比例的和可以小于1
print(rs.get_n_splits(X))
print(rs)
for train_index, test_index in rs.split(X, y):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# %%
sample = pd.DataFrame({'subject': ['p012', 'p012', 'p014', 'p014', 'p014', 'p024', 'p024', 'p024', 'p024', 'p081'],
                       'classname': ['c5', 'c0', 'c1', 'c5', 'c0', 'c0', 'c1', 'c1', 'c2', 'c6'],
                       'img': ['img_41179.jpg', 'img_50749.jpg', 'img_53609.jpg', 'img_52213.jpg', 'img_72495.jpg',
                               'img_66836.jpg', 'img_32639.jpg', 'img_31777.jpg', 'img_97535.jpg', 'img_1399.jpg']})
x_train_names_all = np.array(sample['img'])
y_train_labels_all = np.array(sample['classname'])

rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
n_fold = 1
for train_indices, test_indices in rs.split(sample):
    print('fold {}/5......'.format(n_fold))
    print("train_indices:", train_indices)
    x_train = x_train_names_all[train_indices, ...]
    print("x_train_names:\n", x_train)
    y_train = y_train_labels_all[train_indices, ...]
    print("y_train:\n", y_train)

    print("test_indices:", test_indices)
    x_test = x_train_names_all[test_indices, ...]
    print("x_test:\n", x_test)
    y_test = y_train_labels_all[test_indices, ...]
    print("y_test:\n", y_test)
    n_fold += 1

# %%
# 2.2、GroupShuffleSplit
# sklearn.model_selection.GroupShuffleSplit作用与ShuffleSplit相同，不同之处在于GroupShuffleSplit先将待划分的样本集按groups分组，再按照分组划分训练集、测试集。
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([3, 3, 3, 2, 3, 2])
groups = np.array([1, 3, 1, 2, 3, 2])  # 根据 groups 进行划分，不根据y
# 分3折，先按groups进行划分为3组（优先）； 再按train_size、test_size划分（可以不满足）： test_size=0.25，那么train_size=0.75
group_shuff = GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
print(group_shuff.get_n_splits(X, y, groups))
print(group_shuff)
for train_index, test_index in group_shuff.split(X, y, groups):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

# %%
sample = pd.DataFrame({'subject': ['p012', 'p012', 'p014', 'p014', 'p014', 'p024', 'p024', 'p024', 'p024', 'p081'],
                       'classname': ['c5', 'c0', 'c1', 'c5', 'c0', 'c0', 'c1', 'c1', 'c2', 'c6'],
                       'img': ['img_41179.jpg', 'img_50749.jpg', 'img_53609.jpg', 'img_52213.jpg', 'img_72495.jpg',
                               'img_66836.jpg', 'img_32639.jpg', 'img_31777.jpg', 'img_97535.jpg', 'img_1399.jpg']})
x_train_names_all = np.array(sample['img'])
y_train_labels_all = np.array(sample['classname'])
driver_ids = sample['subject']
_, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
n_fold = 1
rs = GroupShuffleSplit(n_splits=4, test_size=0.25, random_state=0)
for train_indices, test_indices in rs.split(x_train_names_all, y_train_labels_all, groups=driver_indices):
    print('fold {}/4......'.format(n_fold))
    print("train_indices:", train_indices)
    x_train = x_train_names_all[train_indices, ...]
    print("x_train_names:\n", x_train)
    y_train = y_train_labels_all[train_indices, ...]
    print("y_train:\n", y_train)

    print("test_indices:", test_indices)
    x_test = x_train_names_all[test_indices, ...]
    print("x_test:\n", x_test)
    y_test = y_train_labels_all[test_indices, ...]
    print("y_test:\n", y_test)
    n_fold += 1

# %%
# 2.3、StratifiedShuffleSplit 把数据集打乱顺序，然后划分测试集和训练集，
# 训练集额和测试集的比例随机选定，训练集和测试集的比例的和可以小于1,但是还要保证训练集中各类别所占的比例是一样的
X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
y = np.array([1,2,1,2,1,2])
# 分3折，先保证训练集中各类别所占的比例是一样的（优先），再按再按train_size、test_size划分（可以不满足）
sss = StratifiedShuffleSplit(n_splits=3,train_size=.75,test_size=.2,random_state=10) # random_state保证每次运行划分都相同
print(sss.get_n_splits(X,y)) # 折数
print(sss) # 对象
print()
for train_index,test_index in sss.split(X,y):
    print("Train Index:",train_index,",Test Index:",test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    print("训练集长度：%d，测试长度：%d" % (len(y_train), len(y_test)))
    print("-"*30)
    train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
    print("训练集类别：%s，训练集类别数量%s，训练集类别占比：%s" % (train_unique_label, train_counts_label, train_counts_label / len(y_train)))
    test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)
    print("测试集类别：%s，测试集类别数量%s，测试集类别占比：%s" % (test_unique_label, test_counts_label, test_counts_label / len(y_test)))
    print("="*30)
#    break

# In[3]:
# 3、留一法：
# 3.1、LeaveOneGroupOut










