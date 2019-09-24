import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
x = digits.data
y = digits.target


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.neighbors import KNeighborsClassifier



# 1、无交叉验证：
'''
best_score, best_p, best_k = 0, 0 ,0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p) # 定义模型时，手动设定的是 超参数
        knn_clf.fit(x_train, y_train) # 模型实例.fit()训练出一个模型（其实就是得到 模型的参数。注意是参数，不是超参数）
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
best_score, best_p, best_k = 0, 0 ,0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)  # 定义模型时，手动设定的是 超参数
        '''
        一、cross_val_score函数参数：（cross_val_score 只能和 KFold 配合使用，不能和 StratifiedKFold 配合使用）
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

        三、理解：
        单纯从 cross_val_score 来看，每一折交叉验证 模型实例.fit()训练出一个模型（其实就是得到 模型的参数。注意是参数，不是超参数）
        超参数 是在定义模型时就已经设置好的，不由 cross_val_score 交叉验证 控制。
        '''
        kf = KFold(3, shuffle=True)
        # accuracy 准确率指标。当样本比例不平衡时，该指标不准确。
        score = cross_val_score(knn_clf, x_train, y_train, scoring='accuracy', cv=kf).mean() # 3折交叉验证，有3个准确度分数；注意：这里求了mean()
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



# 3、使用交叉验证（使用 网格搜索）：同时得到：1、最优模型参数、2、最优模型的超参数
'''
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# estimator：创建的对象，如上的knn_clf
# param_grid：是一个列表，列表里是算法对象的超参数的取值，用字典存储
# n_jobs：使用电脑的CPU个数，-1代表全部使用
# verbose：每次CV时输出的格式
# cv：
# None：默认参数，函数会使用默认的3折交叉验证
# 整数k：k折交叉验证。对于分类任务，使用StratifiedKFold（类别平衡，每类的训练集占比一样多，具体可以查看官方文档）。对于其他任务，使用KFold
# 交叉验证生成器：得自己写生成器，头疼，略
# 可以生成训练集与测试集的迭代器：同上，略

knn_clf = KNeighborsClassifier()
# param_grid 为模型的超参数，不是参数。
param_grid = [
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(2,11)],
        'p':[i for i in range(1,6)]
    }
]

# GridSearchCV 可以和 StratifiedKFold 配合使用
sKFold = StratifiedKFold(n_splits=3, shuffle = True)
grid_search = GridSearchCV(knn_clf, param_grid, verbose=1, cv=sKFold)
# 这样测试就为了观测 使用 最优模型 再训练数据 的情况
grid_search.fit(x_train[0:600], y_train[0:600]) # 网格搜索 同时得到：1、最优模型参数、2、最优模型的超参数

print(grid_search.best_score_)
print(grid_search.best_params_)
best_knn_clf = grid_search.best_estimator_ # 最佳分类器
print(best_knn_clf)
testScore = best_knn_clf.score(x_test, y_test)
print("GridSearchCV测试结果：", testScore)


best_knn_clf.fit(x_train[600:1078], y_train[600:1078])
print(best_knn_clf) # 可以看出 最优模型的超参数保留，改变的只是最优模型的参数
print("GridSearchCV测试结果：", best_knn_clf.score(x_test, y_test))
'''