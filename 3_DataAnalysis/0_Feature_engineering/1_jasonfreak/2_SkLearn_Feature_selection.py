from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


iris = load_iris()
print(iris.data[0:10])
# print(iris.target[0:10])


# 3、特征选择

# 3.1、Filter

# 3.1.1、方差选择法
from sklearn.feature_selection import VarianceThreshold
# 方差选择法，返回值为特征选择后的数据
# 参数threshold为方差的阈值
vdata = VarianceThreshold(threshold=3).fit_transform(iris.data)
print(vdata[0:10]) # 选择了第三列（方差大于3）
print(np.square(np.std(iris.data[:,0])), np.square(np.std(iris.data[:,1])),
      np.square(np.std(iris.data[:,2])), np.square(np.std(iris.data[:,3])))
