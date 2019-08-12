import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
x = digits.data
y = digits.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

from sklearn.neighbors import KNeighborsClassifier



# 1、无交叉验证：
'''
best_score, best_p, best_k = 0, 0 ,0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        knn_clf.fit(x_train, y_train)
        score = knn_clf.score(x_test, y_test)
        if score > best_score:
            best_score, best_p, best_k = score, p, k

print("Best K =", best_k)
print("Best P =", best_p)
print("Best Score =", best_score)
'''



# 2、使用交叉验证：
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
'''
KFold 测试：
1、设置shuffle=True时，运行两次，发现多次运行的结果不同。
2、设置shuffle=True和random_state=整数，发现每次运行的结果都相同
data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(n_splits=3, shuffle = True, random_state=42)
for train, test in kfold.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))
'''
knn_clf = KNeighborsClassifier()
best_score, best_p, best_k = 0, 0 ,0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        '''
        一、cross_val_score函数参数：
        1、scoring参数：
        1.1、一般来说平均方差(Mean squared error)会用于判断回归(Regression)模型的好坏
        scoring='mean_squared_error'
        1.2、平均准确率
        scoring='accuracy
        2、cv参数：
        cv=3：分成3份做交叉验证，2份做训练，1份做验证
        交叉验证接口sklearn.model_selection.cross_validate没有数据shuffle功能，所以一般结合Kfold一起使用，
        KFold是做分割数据作用，相当于cross_val_score的cv参数；做交叉验证是cross_val_score。

        二、KFold函数参数：
        1、设置shuffle=True时，多次运行的结果不同。
        2、设置shuffle=True和random_state=整数，发现每次运行的结果都相同
        '''
        kf = KFold(3, shuffle=True).get_n_splits(x_train)
        # accuracy 准确率指标。当样本比例不平衡时，该指标不准确。
        score = cross_val_score(knn_clf, x_train, y_train, scoring='accuracy', cv=kf).mean()
        if score > best_score:
            best_score, best_p, best_k = score, p, k

print("Best K =", best_k) # 2
print("Best P =", best_p) # 3
print("Best Score =", best_score) # 0.9833452164435345

# 测试：
knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=2, p=3)
knn_clf.fit(x_train, y_train)
testScore = knn_clf.score(x_test, y_test)
print("cross_val_score测试结果：", testScore)



# 3、使用交叉验证（使用 网格搜索）：
'''
from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()
param_grid = [
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(2,11)],
        'p':[i for i in range(1,6)]
    }
]

grid_search = GridSearchCV(knn_clf, param_grid, verbose=1, cv=3)
grid_search.fit(x_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
best_knn_clf = grid_search.best_estimator_ # 最佳分类器
testScore = best_knn_clf.score(x_test, y_test)
print("GridSearchCV测试结果：", testScore)
'''