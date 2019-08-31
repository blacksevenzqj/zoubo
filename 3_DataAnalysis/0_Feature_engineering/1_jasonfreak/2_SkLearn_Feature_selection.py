from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


iris = load_iris()
print(iris.data[0:5])
# print(iris.target[0:5])
print(set(iris.target))


# 3、特征选择

# 3.1、Filter

# 3.1.1、方差选择法
from sklearn.feature_selection import VarianceThreshold
# 方差选择法，返回值为特征选择后的数据
# 参数threshold为方差的阈值
vdata = VarianceThreshold(threshold=3).fit_transform(iris.data)
# print(vdata[0:10]) # 选择了第三列（方差大于3）
# print(np.square(np.std(iris.data[:,0])), np.square(np.std(iris.data[:,1])),
#       np.square(np.std(iris.data[:,2])), np.square(np.std(iris.data[:,3])))


# ============================================================================================================


'''
A、单变量特征选取：
单变量特征提取的原理是分别计算每个特征的某个统计指标，根据该指标来选取特征。 
SelectKBest、SelectPercentile，前者选择排名前k个的特征，后者选择排名在前k%的特征。选择的统计指标需要指定，
对于regression问题，使用f_regression指标;对于classification问题，可以使用chi2或者f_classif指标。
'''

# 3.1.2、相关系数法
# 计算各个特征对目标值的相关系数以及相关系数的P值
from sklearn.feature_selection import SelectKBest # A、单变量特征选取
from scipy.stats import pearsonr
# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为 计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
# 第一个参数为 k为选择的特征个数
sdata = SelectKBest(lambda X, Y: list(np.array([pearsonr(x, Y) for x in X.T]).T), k=3).fit_transform(iris.data, iris.target)
# print(sdata[0:5])

'''
关于 皮尔森相似度 计算的说明：
1、自变量（连续）之间的 pearson相似度分析；
2、自变量（连续） 与 因变量（连续）之间的 pearson相关度分析：鸢尾花数据的 因变量Y 是 分类变量，只不过是用数值表示的（普通转换）。
这种情况下把 因变量Y 当做是 连续变量 进行皮尔森相似度计算（是否需要将 因变量Y 做WOE转换？）。
'''
# 3.1.2.1、pearsonr皮尔森相关系数矩阵的测试：
# 3.1.2.1.1、pd.DataFrame格式：
idatap = pd.DataFrame(iris.data)
# print(idatap[0:5])
idatap[4] = iris.target
# print(idatap[0:5])
# 从结果可以看到：自变量X 和 因变量Y 的 相关系数 排序为 3>2>0，和SelectKBest函数一致。
# print(idatap.corr(method='pearson'))

# 3.1.2.1.1.1、pd.DataFrame格式：（单独两个特征进行分析）
# min_periods : int, optional，指定每列所需的最小观察数，可选，目前只适合用在pearson和spearman方法。
# print(idatap[[0,4]].corr(method='pearson', min_periods=1)) # 基于DataFrame的皮尔森相关系数矩阵


# 3.1.2.1.2、np.array格式：（的确绕）
'''
首先要明确：
1、np.corrcoef(矩阵A) 只能进行 两个特征之间 的皮尔森相关度计算
2、np.corrcoef(矩阵A)中 矩阵A 必须是：行是特征，列是样本
'''
# 3.1.2.1.2.1、以 np.vstack：按垂直方向（行顺序）堆叠数组构成一个新的数组
ixh = iris.data[:, 0] # 特征 行格式
iyh = iris.target # 因变量 行格式
idatal = np.vstack((ixh, iyh)) # 垂直方向（行顺序）堆叠，正好符合corrcoef函数要求
print(idatal[0:5])
print(np.corrcoef(idatal)) # 行是特征，列是样本

# 3.1.2.1.2.2、以 np.hstack：按水平方向（列顺序）堆叠数组构成一个新的数组
# 如果使用 水平方向（列顺序）堆叠：必须两个矩阵都是二维，才能堆叠出正确格式
# 3.1.2.1.2.2.1、将 因变量Y 按水平方向 堆叠进矩阵，方便其余特征分别与因变量Y进行计算
iY2w = iris.target[np.newaxis, :] # 一维扩展为二维，行格式
idatah = np.hstack((iris.data, iY2w.T)) # 注意，Y做转置T，转置为列
print(idatah[0:5])
idatahf = idatah[:,[0,4]] # 选择 矩阵0列（特征X1）和 矩阵4列（因变量Y）
print(idatahf[0:5])
print(np.corrcoef(idatahf.T)) # .T 转置为 行是特征，列是样本

# 3.1.2.1.2.2.2、单独拿出一个特征进行计算：
iX02w = iris.data[:, 0][np.newaxis, :] # 一维扩展为二维，行格式
iY2w = iris.target[np.newaxis, :] # 一维扩展为二维，行格式
idatah = np.hstack((iX02w.T, iY2w.T)) # .T 转置为 列格式 合并
print(idatah[0:5])
print(np.corrcoef(idatah.T)) # .T 转置为 行是特征，列是样本

# -------------------------------------------------------------------------------------------------------------

# 3.1.3、方检验
# 经典的卡方检验是检验 定性自变量（分类） 对 定性因变量（分类） 的相关性。假设自变量有N种取值，因变量有M种取值，
# 考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量：



