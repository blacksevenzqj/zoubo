# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
from scipy import stats

# 1、SVM可视化
# In[]:
X,y = make_blobs(n_samples=50, centers=2, random_state=0,cluster_std=0.6)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow") # c颜色类别，rainbow彩虹色
plt.xticks([])
plt.yticks([])
plt.show()

# In[]:
# 首先要有散点图
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca() #获取

# In[]:
# 获取平面上两条坐标轴的最大值和最小值
xlim = ax.get_xlim()
ylim = ax.get_ylim()
 
# 在最大值和最小值之间形成30个规律的数据
axisx = np.linspace(xlim[0],xlim[1],30)
axisy = np.linspace(ylim[0],ylim[1],30)
 
axisy,axisx = np.meshgrid(axisy,axisx)
# 我们将使用这里形成的二维数组作为我们contour函数中的X和Y
# 使用meshgrid函数将两个一维向量转换为特征矩阵
# 核心是将两个特征向量广播，以便获取y.shape * x.shape这么多个坐标点的横坐标和纵坐标
 
xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
# 其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
# xy就是已经形成的网格，它是遍布在整个画布上的密集的点
 
plt.scatter(xy[:,0],xy[:,1],s=1,cmap="rainbow")

# In[]:
# 建模，通过fit计算出对应的决策边界
clf = SVC(kernel = "linear").fit(X,y) # 计算出对应的决策边界
Z = clf.decision_function(xy).reshape(axisx.shape)
# 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
# 然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致

# 首先要有散点图
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca() # 获取当前的子图，如果不存在，则创建新的子图
# 画决策边界和平行于决策边界的超平面
ax.contour(axisx,axisy,Z
           ,colors="k"
           ,levels=[-1,0,1] # 画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
           ,alpha=0.5 # 透明度
           ,linestyles=["--","-","--"])
 
ax.set_xlim(xlim) # 设置x轴取值
ax.set_ylim(ylim)

# In[]:
# 记得Z的本质么？是输入的样本到决策边界的距离，而contour函数中的level其实是输入了这个距离
# 让我们用一个点来试试看
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.scatter(X[10,0],X[10,1],c="black",s=50,cmap="rainbow")

# In[]:
temp = clf.decision_function(X[10].reshape(1,2)) # -3.33917354
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
ax = plt.gca()
ax.contour(axisx,axisy,Z
            ,colors="k"
            ,levels=temp
            ,alpha=0.5
            ,linestyles=["--"])

# In[]:
# 将上述过程包装成函数：
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x) 
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P, colors="k", levels=[-1,0,1], alpha=0.5, linestyles=["--","-","--"]) 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
 
# 则整个绘图过程可以写作：
clf = SVC(kernel = "linear").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)

# In[]:
clf.predict(X)
# 根据决策边界，对X中的样本进行分类，返回的结构为n_samples
 
clf.score(X,y)
# 返回给定测试数据和标签的平均准确度
 
clf.support_vectors_
# 返回支持向量坐标
 
clf.n_support_#array([2, 1])
# 返回每个类中支持向量的个数

# In[]:
clf = SVC(kernel = "linear").fit(X,y)
sv_list = clf.support_vectors_
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.scatter(sv_list[:,0],sv_list[:,1],c="black",s=50,cmap="rainbow")
plot_svc_decision_function(clf)



# In[]:
# 2、非线性数据，核函数
from sklearn.datasets import make_circles
X,y = make_circles(100, factor=0.1, noise=.1)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plt.show()

# In[]:
clf = SVC(kernel = "linear").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)
clf.score(X,y)

# In[]:
#定义一个由x计算出来的新维度r
r = np.exp(-(X**2).sum(axis=1))
 
rlim = np.linspace(min(r),max(r),100)
 
# 定义一个绘制三维图像的函数
# elev表示上下旋转的角度
# azim表示平行旋转的角度
def plot_3D(elev=30,azim=30,X=X,y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()
    
plot_3D()

# In[]:
# 如果放到jupyter notebook中运行（在 jupyter notebook 可以交互式操作）
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
 
from sklearn.datasets import make_circles
X,y = make_circles(100, factor=0.1, noise=.1)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
 
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x) 
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
 
clf = SVC(kernel = "linear").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)
 
r = np.exp(-(X**2).sum(1))
 
rlim = np.linspace(min(r),max(r),0.2)
 
 
def plot_3D(elev=30,azim=30,X=X,y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()
 
from ipywidgets import interact,fixed
interact(plot_3D,elev=[0,30,60,90],azip=(-180,180),X=fixed(X),y=fixed(y))
plt.show()

# In[]:
# 非线性核函数： rbf
clf = SVC(kernel = "rbf").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
plot_svc_decision_function(clf)



# In[]:
# 3、探索核函数在不同数据集上的表现
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm # from sklearn.svm import SVC  两者都可以
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification

# In[]:
n_samples = 100
 
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=n_samples, centers=2, random_state=5), # 分簇的数据集
    # n_features：特征数， n_informative：带信息的特征数， n_redundant：不带信息的特征数-噪音； 用于测试特征选择、 PCA效果。
    make_classification(n_samples=n_samples,n_features=2,n_informative=2,n_redundant=0,random_state=5)
    ]
 
Kernel = ["linear","poly","rbf","sigmoid"]
 
#四个数据集分别是什么样子呢？
for X,Y in datasets:
    plt.figure(figsize=(5,4))
    plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="rainbow")

# In[]:
[*enumerate(datasets)] == list(enumerate(datasets)) #  enumerate、map、zip都可以这样展开
[*enumerate(datasets)]
# index，(X,Y) = [(索引, array([特矩阵征X],[标签Y]))]
# In[]:
nrows=len(datasets)
ncols=len(Kernel) + 1
 
fig, axes = plt.subplots(nrows, ncols,figsize=(20,16))

#第一层循环：在不同的数据集中循环
for ds_cnt, (X,Y) in enumerate(datasets):
    
    #在图像中的第一列，放置原数据的分布
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    
    #第二层循环：在不同的核函数中循环
    #从图像的第二列开始，一个个填充分类结果
    for est_idx, kernel in enumerate(Kernel):
        
        #定义子图位置
        ax = axes[ds_cnt, est_idx + 1]
        
        #建模
        clf = svm.SVC(kernel=kernel, gamma=2).fit(X, Y)
        score = clf.score(X, Y)
        
        #绘制图像本身分布的散点图
        ax.scatter(X[:, 0], X[:, 1], c=Y
                   ,zorder=10
                   ,cmap=plt.cm.Paired,edgecolors='k')
        #绘制支持向量
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                    facecolors='none', zorder=10, edgecolors='white') # facecolors='none':透明的
        
        #绘制决策边界
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        
        #np.mgrid，合并了我们之前使用的np.linspace和np.meshgrid的用法
        #一次性使用最大值和最小值来生成网格
        #表示为[起始值：结束值：步长]
        #如果步长是复数，则其整数部分就是起始值和结束值之间创建的点的数量，并且结束值被包含在内
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        #np.c_，类似于np.vstack的功能
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        #填充等高线不同区域的颜色
        ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        #绘制等高线
        ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-1, 0, 1])
        
        #设定坐标轴为不显示
        ax.set_xticks(())
        ax.set_yticks(())
        
        #将标题放在第一行的顶上
        if ds_cnt == 0:
            ax.set_title(kernel)
            
        #为每张图添加分类的分数   
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0') # 不显示小数的整数位0
                , size=15
                , bbox=dict(boxstyle='round', alpha=0.8, facecolor='white') # 为分数添加一个白色的格子作为底色
                , transform=ax.transAxes # 确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                , horizontalalignment='right' # 位于坐标轴的什么方向
               )
 
plt.tight_layout()
plt.show()



# In[]:
# 4、探索核函数的优势和缺陷
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from time import time
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler

# In[]:
data_cancer = load_breast_cancer()
X = data_cancer.data.copy()
y = data_cancer.target.copy()
 
print(X.shape)
print(np.unique(y), set(y))
# In[]: 
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

# In[]:
# PCA_SVD联合降维
from sklearn.decomposition import PCA
pca = PCA(2, whiten = True, svd_solver='auto').fit(X) 
V = pca.components_ # 新特征空间
print(V.shape) #V(k，n) 

X_dr = pca.transform(X) # PCA降维后的信息保存量
print(X_dr.shape) 

plt.scatter(X_dr[:,0],X_dr[:,1],c=y)
plt.show()

# In[]:
# t-SNE降维
from sklearn import manifold
# init='pca'：初始化，默认为random。取值为random为随机初始化，取值为pca为利用PCA进行初始化（常用）
tsne = manifold.TSNE(n_components=2,  init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape) 

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y)
plt.show()

# In[]:
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
# In[]:
'''
1、我们发现，如果使用poly多项式核函数，默认degree=3，使用非线性方式； 怎么跑都跑不出来，模型一直停留在linear线性核函数之后，
就没有再打印结果了。这证明，多项式核函数此时此刻要消耗大量的时间，运算非常的缓慢。
2、排除poly多项式核函数再运行，可以有两个发现。首先，乳腺癌数据集是一个线性数据集，线性核函数跑出来的效果很好。rbf和sigmoid两个
擅长非线性的数据从效果上来看完全不可用。其次，线性核函数的运行速度远远不如非线性的两个核函数。
'''
#Kernel = ["linear","poly","rbf","sigmoid"]
Kernel = ["linear","rbf", "sigmoid"] # poly多项式核函数 默认是degree=3，使用非线性方式，计算不出来
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel
             , gamma="auto"
             , cache_size=5000 # 使用计算的内存，单位是MB，默认是200MB
            ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print(time()-time0)
# In[]:
'''
1、设置poly多项式核函数degree=1，使用线性方式，的运行速度立刻加快了，并且精度也提升到了接近线性核函数的水平。
但是，我们之前的实验中，我们了解说，rbf在线性数据上也可以表现得非常好，那在这里，为什么跑出来的结果如此糟糕呢？
2、其实，这里真正的问题是数据的量纲问题。回忆一下我们如何求解决策边界，如何判断点是否在决策边界的一边？
是靠计算”距离“，虽然我们不能说SVM是完全的距离类模型，但是它严重受到数据量纲的影响。
'''
Kernel = ["linear","poly","rbf","sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel
             , gamma="auto"
             , degree = 1 # poly多项式核函数 设置degree=1，使用线性方式（>1即使用非线性方式）
             , cache_size=5000
            ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print(time()-time0)

# In[]:
# 数据分布检测：
# 虽然我们不能说SVM是完全的距离类模型，但是它严重受到数据量纲的影响。让我们来探索一下乳腺癌数据集的量纲
data = pd.DataFrame(X.copy())
temp_desc_svm = data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T # 描述性统计
#temp_desc_svm.to_csv("C:\\Users\\dell\\Desktop\\123123\\temp_desc_svm.csv")
'''
从mean列和std列可以看出严重的量纲不统一
从1%的数据和最小值相对比，90%的数据和最大值相对比，查看是否是正态分布或偏态分布，如果差的太多就是偏态分布，谁大方向就偏向谁
可以发现数据大的特征存在偏态问题
这个时候就需要对数据进行标准化
'''

# In[]:
# 正太分布检测：
'''
原假设：样本来自一个正态分布的总体。
备选假设：样本不来自一个正态分布的总体。
w和p同向： w值越小； p-值越小、接近于0； 拒绝原假设。
'''
var = data.columns
shapiro_var = {}
for i in var:
    shapiro_var[i] = stats.shapiro(data[i]) # 返回 w值 和 p值
    
shapiro = pd.DataFrame(shapiro_var).T.sort_values(by=1, ascending=False)

fig, axe = plt.subplots(1,1,figsize=(15,10))
axe.bar(shapiro.index, shapiro[0], width=.4) #自动按X轴---skew.index索引0-30的顺序排列
#在柱状图上添加数字标签
for a, b in zip(shapiro.index, shapiro[0]):
    # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)

# In[]:
# 数据偏度检测：
var = data.columns
skew_var = {}
for i in var:
    skew_var[i] = abs(data[i].skew())
    
skew = pd.Series(skew_var).sort_values(ascending=False)

fig, axe = plt.subplots(1,1,figsize=(15,10))
axe.bar(skew.index, skew, width=.4) #自动按X轴---skew.index索引0-30的顺序排列

#在柱状图上添加数字标签
for a, b in zip(skew.index, skew):
    # a是X轴的柱状体的索引， b是Y轴柱状体高度， '%.4f' % b 是显示值
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=12)

# In[]:
var_x_ln = skew.index[skew > 0.9] # skew的索引 --- data的列名
print(var_x_ln, len(var_x_ln))
# In[]:
fig, axe = plt.subplots(len(var_x_ln),1,figsize=(20,28*6))

for i,var in enumerate(var_x_ln):
    sns.distplot(data[var], bins=100, color='green', ax=axe[i])
    axe[i].set_title('feature: ' + str(var))
    axe[i].set_xlabel('')

# In[]:
# 将偏度大于1的连续变量 取对数
for i in var_x_ln:
    if min(data[i]) <= 0:
        data[i] =np.log(data[i] + abs(min(data[i])) + 0.01) # 负数取对数的技巧
    else:
        data[i] =np.log(data[i])

X = data.values

# In[]:
# 数据标准化：
#X = StandardScaler().fit_transform(X) # 将数据转化为0,1正态分布
#data = pd.DataFrame(X)
        
X = StandardScaler().fit_transform(X)
data = pd.DataFrame(X)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
'''
标准化之后，从 “4、探索核函数的优势和缺陷” 开始处从新进行数据分布可视化测试：
量纲统一之后，可以观察到，所有核函数的运算时间都大大地减少了，尤其是对于线性核来说，而多项式核函数居
然变成了计算最快的。其次，rbf表现出了非常优秀的结果。经过我们的探索，我们可以得到的结论是：
1、poly多项式核函数 在 高次项degree > 1时（非线性） 计算非常缓慢。
2、rbf 和 多项式核函数都不擅长处理量纲不统一的数据集。
幸运的是，这两个缺点都可以由数据无量纲化来解决。因此，SVM执行之前，非常推荐先进行数据的无量纲化！

对比图在 “SVM1代码图” 文件夹中： 
3.1.1、标准化： 不会改变数据分布，数据散点图、偏度、直方图、正太检验 都没有大的变化。
3.1.2、标准化： 改变了 PCA、t-SNE、SVM 的计算结果，因为它们都需要无量纲化处理。
3.2.1、偏态数据取log： 改变数据分布，数据散点图、偏度、直方图、正太检验 都有变化。
3.2.2、偏态数据取log： 改变了 PCA、t-SNE 的结果，但是正确性不好。
3.3.1、偏态数据取log 再 标准化： 改变数据分布，数据散点图、偏度、直方图、正太检验 都有变化（偏态数据取log的作用）
3.3.2、偏态数据取log 再 标准化： 改变了 PCA、t-SNE、SVM 的计算结果，因为它们都需要无量纲化处理（标准化的作用）

4、最后得到2组数据： 
4.1、不做偏度，做标准化： 
4.2、做偏度，做标准化： linear分低于4.1， poly和sigmoid分高于4.1， rbf分和4.1相同
'''
# In[]:
# 标准化后再进行验证：
Kernel = ["linear","poly","rbf","sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf= SVC(kernel = kernel
             , gamma="auto"
             , degree = 1 # poly多项式核函数 设置degree=1，使用线性方式（>1即使用非线性方式）
             , cache_size=5000
            ).fit(Xtrain,Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel,clf.score(Xtest,Ytest)))
    print(time()-time0)


# In[]:
# 5、选取与核函数相关的参数：degree & gamma & coef0
# 5.1、rbf高斯核函数
score = []
gamma_range = np.logspace(-10, 1, 50) # 等比数列： 返回在对数刻度上均匀间隔的数字
for i in gamma_range:
    clf = SVC(kernel="rbf",gamma = i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), gamma_range[score.index(max(score))]) # 0.9707602339181286 0.020235896477251554
plt.plot(gamma_range,score)
plt.show()
# In[]:
# 细化gamma取值区间：
score = []
gamma_range = np.logspace(-5, 1, 50)
for i in gamma_range:
    clf = SVC(kernel="rbf",gamma = i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), gamma_range[score.index(max(score))]) # 0.9766081871345029 0.026826957952797246
rbf_gamma = gamma_range[score.index(max(score))]
plt.plot(gamma_range,score)
plt.show()

# In[]:
# 5.2、poly多项式核函数
from sklearn.model_selection import StratifiedShuffleSplit # 用于支持带交叉验证的网格搜索
from sklearn.model_selection import GridSearchCV # 带交叉验证的网格搜索
 
time0 = time()

# 运行次数1： gamma_range=50 * coef0_range=10 = 500次
gamma_range = np.logspace(-10,1,50) # 等比数列
coef0_range = np.linspace(0,5,10) # 等差数列
param_grids = dict(gamma = gamma_range, coef0 = coef0_range)

# 运行次数2： 500次 * n_splits=5 = 2500次； 
# 其中random_state设置了定值，也就是这500次分别进行交叉验证时，每一次切分的数据集都是相同的。
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=420) 

grid = GridSearchCV(SVC(kernel = "poly", degree=1, cache_size=5000), param_grid=param_grids, cv=cv)
grid.fit(X, y)
 
print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
print(time()-time0)



# In[]:
# 6、硬间隔与软间隔：重要参数C
# 调线性核函数
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = SVC(kernel="linear",C=i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()

# In[]:
# 使用rbf
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = SVC(kernel="rbf", C=i, gamma = rbf_gamma, cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), C_range[score.index(max(score))]) # 0.9883040935672515 9.80265306122449
plt.plot(C_range,score)
plt.show()
# In[]:
#进一步细化
score = []
C_range = np.linspace(5,15,50)
for i in C_range:
    clf = SVC(kernel="rbf", C=i, gamma = rbf_gamma, cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
    
print(max(score), C_range[score.index(max(score))]) # 0.9883040935672515 9.89795918367347
rbf_c = C_range[score.index(max(score))]
plt.plot(C_range,score)
plt.show()
