# coding:utf-8

import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt

alpha = 0.5
lr = 20

data = cv2.imread('lena.png')
line = np.random.uniform(-1,1,3)
line /= np.sqrt(line[0] ** 2 + line[1] ** 2)
height, width = data.shape[0], data.shape[1]
data_pad = np.zeros(shape=(height+2, width+2, 3))
data_pad[1:height+1, 1:width+1, :] = data

a, b = np.mgrid[0:height, 0:width]
result = np.zeros(shape=(height, width, 3))
t_start = time()

d = line[0] * a + line[1] * b + line[2]
d = (d - d.min()) / (d.max() - d.min())
w1 = alpha / (d + alpha)
w2 = 1 - d ** alpha
w = np.fabs(w1 + w2)

ii = (a - lr * w * line[1]).clip(-1, height).astype(np.int)
jj = (b - lr * w * line[0]).clip(-1, width).astype(np.int)
result = data_pad[ii+1,jj+1]
t_end = time()
print('耗时：', t_end - t_start, '秒')
cv2.imwrite('result.png', result)

im = cv2.imread('result.png')
cv2.imshow('Show', im)
cv2.waitKey(0)
