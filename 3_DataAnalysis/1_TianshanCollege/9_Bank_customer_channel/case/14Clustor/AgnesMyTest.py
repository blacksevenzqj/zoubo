import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np
import matplotlib.pylab as plt


#生成待聚类的数据点,这里生成了20个点,每个点4维:
# points = scipy.randn(20,4)
np.random.seed(2)
# points = np.random.rand(3, 3)
points = np.random.randint(0,10,(3,2))
print(points)


j01 = np.sqrt(np.sum(np.square(points[0]-points[1])))
j02 = np.sqrt(np.sum(np.square(points[0]-points[2])))
j12 = np.sqrt(np.sum(np.square(points[1]-points[2])))
print(j01, j02, j12)

print(np.sum([j01,j02]) / 2)
print(np.sum([j01,j12]) / 2)
print(np.sum([j02,j12]) / 2)


# w01_02 = (j01 - (j01 + j02) / 2)**2 + (j02 - (j01 + j02) / 2)**2
# w01_12 = (j01 - (j01 + j12) / 2)**2 + (j12 - (j01 + j12) / 2)**2
# w02_12 = (j02 - (j02 + j12) / 2)**2 + (j12 - (j02 + j12) / 2)**2
# print(w01_02, w01_12, w02_12)

#1. 层次聚类
#生成点与点之间的距离矩阵,这里用的欧氏距离:
disMat = sch.distance.pdist(points,'euclidean')
print("距离矩阵？？？：", disMat)


#进行层次聚类:
Z = sch.linkage(disMat,method='average') # ward、average
print(Z)

# 直接传入 坐标点数据（矩阵）
Z = sch.linkage(points, metric='euclidean', method='average')
print(Z)

#将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
P = sch.dendrogram(Z)
# plt.savefig('plot_dendrogram.png')
# #根据linkage matrix Z得到聚类结果:
cluster= sch.fcluster(Z, t=0)
print(cluster)