import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold

# 行为样本，列为特征
X = np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])

y = np.array([1,1,0,0,1,1,0,0])


floder = KFold(n_splits=4, random_state=0, shuffle=False)
for train_index, test_index in floder.split(X,y):
    print('Train: %s | test: %s' % (train_index, test_index))
    # print("Train_X: %s"  %  X[train_index])
    # print("Train_y: %s" % y[train_index])
    # print("test_X: %s" % X[test_index])
    # print("test_y: %s" % y[test_index])
    print(" ")

print("----------------------------------------------------------------------------------------------------------")

# StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
sfolder = StratifiedKFold(n_splits=4, random_state=0, shuffle=False)
for train_index, test_index in sfolder.split(X,y):
    print('Train: %s | test: %s' % (train_index, test_index))
    print(" ")


print("----------------------------------------------------------------------------------------------------------")


# stratifiedKFold：保证训练集中每一类的比例是相同的（尽量）
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([1, 1, 1, 2, 2, 2])
skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(X, y)
print(skf)
for train_index, test_index in skf.split(X, y):
    print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

print()
print(X_train)
print(X_test)
print(y_train)
print(y_test)