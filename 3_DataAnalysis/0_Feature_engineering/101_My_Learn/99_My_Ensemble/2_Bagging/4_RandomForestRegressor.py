from sklearn.datasets import load_boston  # 一个标签是连续西变量的数据集
from sklearn.model_selection import cross_val_score  # 导入交叉验证模块
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归系

boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)  # 实例化
regressor.fit(boston.data, boston.target)
regressor.score(boston.data, boston.target)  # R方


# 如果不写 neg_mean_squared_error，回归评估默认是R平方
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)  # 实例化
scores = cross_val_score(regressor, boston.data, boston.target, cv=10
                         , scoring="neg_mean_squared_error"  # 负最小均方差
                         )