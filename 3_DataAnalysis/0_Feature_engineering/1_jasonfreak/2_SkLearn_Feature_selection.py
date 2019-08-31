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


# -------------------------------------------------------------------------------------------------------------


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
关于 皮尔森相似度 计算的说明：（连续-连续）
1、自变量（连续）之间的 pearson相似度分析；
2、自变量（连续） 与 因变量（连续）之间的 pearson相关度分析：鸢尾花数据的 因变量Y 是 分类变量，是用数值表示的（普通转换）。
这种情况下把 因变量Y 当做是 连续变量 进行皮尔森相似度计算（是否需要将 因变量Y 做WOE转换？
之前做的是 自变量X（分类）WOE转换 为 连续变量 后 做PCA）。
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
# ixh = iris.data[:, 0] # 特征 行格式
# iyh = iris.target # 因变量 行格式
# idatal = np.vstack((ixh, iyh)) # 垂直方向（行顺序）堆叠，正好符合corrcoef函数要求
# print(idatal[0:5])
# print(np.corrcoef(idatal)) # 行是特征，列是样本

# 3.1.2.1.2.2、以 np.hstack：按水平方向（列顺序）堆叠数组构成一个新的数组
# 如果使用 水平方向（列顺序）堆叠：必须两个矩阵都是二维，才能堆叠出正确格式
# 3.1.2.1.2.2.1、将 因变量Y 按水平方向 堆叠进矩阵，方便其余特征分别与因变量Y进行计算
# iY2w = iris.target[np.newaxis, :] # 一维扩展为二维，行格式
# idatah = np.hstack((iris.data, iY2w.T)) # 注意，Y做转置T，转置为列
# print(idatah[0:5])
# idatahf = idatah[:,[0,4]] # 选择 矩阵0列（特征X1）和 矩阵4列（因变量Y）
# print(idatahf[0:5])
# print(np.corrcoef(idatahf.T)) # .T 转置为 行是特征，列是样本

# 3.1.2.1.2.2.2、单独拿出一个特征进行计算：
# iX02w = iris.data[:, 0][np.newaxis, :] # 一维扩展为二维，行格式
# iY2w = iris.target[np.newaxis, :] # 一维扩展为二维，行格式
# idatah = np.hstack((iX02w.T, iY2w.T)) # .T 转置为 列格式 合并
# print(idatah[0:5])
# print(np.corrcoef(idatah.T)) # .T 转置为 行是特征，列是样本

# -------------------------------------------------------------------------------------------------------------

# 3.1.3、方检验（分类-分类）
# 经典的卡方检验是检验 定性自变量（分类） 对 定性因变量（分类） 的相关性。假设自变量有N种取值，因变量有M种取值，
# 考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量：
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# # 选择K个最好的特征，返回选择特征后的数据
# kfdata = SelectKBest(chi2, k=3).fit_transform(iris.data, iris.target)
# print(kfdata[0:5]) # 卡方检验计算结果 和 皮尔森相似度计算结果相同

# -------------------------------------------------------------------------------------------------------------

# 3.1.4、互信息法（分类-分类）包装不起来，暂时不管了
# from sklearn.feature_selection import SelectKBest
# from minepy import MINE
# #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
# #选择K个最好的特征，返回特征选择后的数据
# SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)


# ********************************************************************************************************


'''
B、循环特征选取：
不单独地检验某个特征的价值，而是检验特征集的价值。对于一个数量为n的特征集合，子集的个数为2的n次方减一。
通过指定一个学习算法，通过算法计算所有子集的error，选择error最小的子集作为选取的特征。
1、对初始特征集合中每个特征赋予一个初始权重。
2、训练，将权重最小的特征移除。
3、不断迭代，直到特征集合的数目达到预定值。
'''

# 3.2、Wrapper

# 3.2.1、递归特征消除法
# 递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
# 使用feature_selection库的RFE类来选择特征的代码如下：
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
refdata = RFE(estimator=LogisticRegression(), n_features_to_select=3).fit_transform(iris.data, iris.target)
# print(refdata[0:5]) # 这里选择了 1、2、3特征，和上面的方法结果有区别。


# ============================================================================================================


# 3.3、Embedded

# 3.3.1 基于惩罚项的特征选择法
# 使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
# 使用feature_selection库的SelectFromModel类结合带L1惩罚项的逻辑回归模型，来选择特征的代码如下：
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
#带L1惩罚项的逻辑回归作为基模型的特征选择
sl1data = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
# print(sl1data[0:5])

# L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
# 故，可结合L2惩罚项来优化。具体操作为：若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
# 将这一集合中的特征平分L1中的权值，故需要构建一个新的逻辑回归模型：
from sklearn.linear_model import LogisticRegression

class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self

# 使用feature_selection库的SelectFromModel类结合带L1以及L2惩罚项的逻辑回归模型，来选择特征的代码如下：
from sklearn.feature_selection import SelectFromModel
#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
sl1l2data = SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)
# print(sl1l2data)


# ----------------------------------------------------------------------------------------------------------------------------


# 3.3.2、基于树模型的特征选择法
# 树模型中GBDT也可用来作为基模型进行特征选择，使用feature_selection库的SelectFromModel类结合GBDT模型，来选择特征的代码如下：
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
# GBDT作为基模型的特征选择
sgbdtdata = SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
# print(sgbdtdata[0:5])