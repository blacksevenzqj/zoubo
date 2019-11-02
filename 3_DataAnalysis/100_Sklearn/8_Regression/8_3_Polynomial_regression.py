# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 20:33:45 2019

@author: dell
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[]:
# 方式一： 多项式不生成截距项，截距项由线性回归模型生成。
X = np.arange(1,4).reshape(-1,1)
# include_bias=False 不生成截距项，只生成特征项，截距项让模型生成
xxx = PolynomialFeatures(degree=3,include_bias=False).fit_transform(X)
rnd = np.random.RandomState(42) #设置随机数种子
y = rnd.randn(3)

lr = LinearRegression().fit(xxx,y)
# 生成了多少个系数？
print(lr.coef_)
# 查看截距
print(lr.intercept_)

# In[]:
# 方式二： 多项式生成截距项， 线性回归模型不生成截距项。
X = np.arange(1,4).reshape(-1,1)
# include_bias=False 不生成截距项，只生成特征项，截距项让模型处理
xxx = PolynomialFeatures(degree=3).fit_transform(X)
rnd = np.random.RandomState(42) #设置随机数种子
y = rnd.randn(3)

lr = LinearRegression(fit_intercept=False).fit(xxx,y)
# 生成了多少个系数？
print(lr.coef_)
# 查看截距
print(lr.intercept_) # 0.0

# In[]:
X = np.arange(6).reshape(3, 2)
# 不要截距项， 且只要交互项：interaction_only=True
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit(X)
print(poly.get_feature_names())
xxx = poly.transform(X)



# In[]:
# 多项式拟合sin数据
rnd = np.random.RandomState(42) # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)
y = np.sin(X) + rnd.normal(size=len(X)) / 3

# 将X升维，准备好放入sklearn中
X = X.reshape(-1,1)

# 创建测试数据，均匀分布在训练集X的取值范围内的一千个点
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
# In[]:
d=5

poly = PolynomialFeatures(degree=d, include_bias=False).fit(X) # 训练集多项式数据
X_ = poly.transform(X)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1) # 测试集
line_ = poly.transform(line) # 测试集多项式

LinearR = LinearRegression().fit(X, y)
LinearR_ = LinearRegression().fit(X_, y)
print(LinearR_.intercept_)

# 放置画布
fig, ax1 = plt.subplots(1)

# 将测试数据带入predict接口，获得模型的拟合效果并进行绘制
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green'
         ,label="linear regression")
ax1.plot(line, LinearR_.predict(line_), linewidth=2, color='red'
         ,label="Polynomial regression")

ax1.plot(X[:, 0], y, 'o', c='k') # 训练集
ax1.plot(line, np.sin(line), c='k') # 测试集：  np.sin(line)为测试集Y

#其他图形选项
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Linear Regression ordinary vs poly")
plt.tight_layout()
plt.show()

print(LinearR_.score(line_, np.sin(line)))



# In[]:
# 拟合 加利福尼亚房屋数据集
from sklearn.datasets import fetch_california_housing as fch

housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
housevalue.feature_names
# In[]:
poly = PolynomialFeatures(degree=5, include_bias=False).fit(X)
X_ = poly.transform(X)
print(X_)
# In[]:
# 在这之后，我们依然可以直接建立模型，然后使用线性回归的coef_属性来查看什么特征对标签的影响最大
reg = LinearRegression().fit(X_,y)
print(reg.intercept_)

print([*zip(poly.get_feature_names(),reg.coef_)][:10])
# 放到dataframe中进行排序
coeff = pd.DataFrame([poly.get_feature_names(),reg.coef_.tolist()]).T
coeff.columns = ["feature","coef"]
coeff.sort_values(by="coef", inplace=True)

# In[]:
from time import time
time0 = time()
print("R2:{}".format(reg.score(X_,y)))
print("time:{}".format(time()-time0))


# In[]:
# 假设使用其他模型？
from sklearn.ensemble import RandomForestRegressor as RFR

time0 = time()
print("R2:{}".format(RFR(n_estimators=100).fit(X,y).score(X,y))) # R2:0.9743205003727138
print("time:{}".format(time()-time0))










