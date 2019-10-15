import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target # ndarray
n_samples, n_features = X.shape
print(X.shape, y.shape)
print(set(y))
# print(X[0:5])
# print(y[0:5])
# print(X[0].reshape(8,8))

img = X[1].reshape(8,8) # 一个数字用二进制表示为：8x8的矩阵
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])


'''显示原始数据'''
n = 20  # 每行20个数字，每列20个数字
img = np.zeros((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])

# plt.show()


print("==========================================================================================================")


# Kaggle中是先 标准化之后 再进的 t-SNE；这里直接就进 t-SNE，之后再 归一化
from sklearn import preprocessing

'''t-SNE'''
# init='pca'：初始化，默认为random。取值为random为随机初始化，取值为pca为利用PCA进行初始化（常用）
tsne = manifold.TSNE(n_components=2,  init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print(X.shape, X_tsne.shape) # (1083, 64) (1083, 2)
print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
# 降维之后 进行 归一化
x_min, x_max = X_tsne.min(axis=0), X_tsne.max(axis=0) # [-49.062893 -52.216053] [47.26139 45.00642] 因是2个特征
print(x_min, x_max)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
print(X_norm[0:5])
X_norm = preprocessing.MinMaxScaler().fit_transform(X_tsne)
print(X_norm[0:5])
# X_norm = preprocessing.scale(X_tsne) # 不能标准化
# print(X_norm[0:5])

plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])

plt.show()


print("-----------------------------------------------------------------------------------------------------")


# 不能正常显示
from sklearn.decomposition import PCA, TruncatedSVD

X_z = preprocessing.scale(X) # 标准化
# X_z = preprocessing.StandardScaler().fit_transform(X)
X_reduced_pca = PCA(n_components=2, random_state=501, svd_solver='auto').fit_transform(X_z)

plt.figure(figsize=(8, 8))
for i in range(X_reduced_pca.shape[0]):
    plt.text(X_reduced_pca[i, 0], X_reduced_pca[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])

plt.show()