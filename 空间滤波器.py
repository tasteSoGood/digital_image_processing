# -*-coding: utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functools import reduce
from 椒盐噪声 import salt_noise

class space_filter:
    def __init__(self, pic):
        self.pic = pic
        self.shape = pic.shape
    
    def imfilter(self, w):
        # 矩阵算子形式的滤波器
        temp = self.pic.copy()
        # 加边距
        r, c = np.array(w.shape) // 2
        temp = cv.copyMakeBorder(temp, r, r, c, c, cv.BORDER_REPLICATE)
        # 运算
        res = temp.copy()
        w = w[::-1, ::-1] # 卷积
        for i in range(r, temp.shape[0] - r):
            for j in range(c, temp.shape[1] - c):
                res[i, j] = np.dot(temp[i - r : i + r + 1, j - c : j + c + 1].ravel(), w.ravel().T)
        return res[r : -r, c : -c]

    def func_filter(self, scale, method):
        temp = self.pic.copy()
        # 加边距
        r, c = scale // 2, scale // 2
        temp = cv.copyMakeBorder(temp, r, r, c, c, cv.BORDER_REPLICATE)
        res = temp.copy()
        # 均值滤波器
        if method == 'arithmetic':
            # 算术均值滤波器
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.mean(temp[i - r : i + r + 1, j - c : j + c + 1])

        elif method == 'geometric':
            # 几何均值滤波器
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = reduce(lambda a, b: a * b, temp[i - r : i + r + 1, j - c : j + c + 1].ravel())
            res = res ** (1 / (scale ** 2))

        elif method == 'harmonic':
            # 逆谐波均值滤波器
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = scale * scale / np.sum(1 / temp[i - r : i + r + 1, j - c : j + c + 1])

        elif method == 'inv harmonic':
            # 二阶逆谐波均值滤波器
            q = 2
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.sum(temp[i - r : i + r + 1, j - c : j + c + 1] ** (q + 1)) / np.sum(temp[i - r : i + r + 1, j - c : j + c + 1] ** q)

        # 统计排序滤波器
        elif method == 'median':
            # 中值滤波器
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.median(temp[i - r : i + r + 1, j - c : j + c + 1])

        elif method == 'maximum':
            # 最大值滤波器
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.max(temp[i - r : i + r + 1, j - c : j + c + 1])

        elif method == 'minimum':
            # 最小值滤波器
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.min(temp[i - r : i + r + 1, j - c : j + c + 1])

        elif method == 'middle point':
            # 中点滤波器（最大值和最小值的一半）
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = 0.5 * (np.min(temp[i - r : i + r + 1, j - c : j + c + 1]) + np.min(temp[i - r : i + r + 1, j - c : j + c + 1]))

        elif method == 'alpha':
            # 修正的阿尔法均值滤波器
            d = 2
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    sorted_mat = np.sort(temp[i - r : i + r + 1, j - c : j + c + 1], axis = None)
                    res[i, j] = np.sum(sorted_mat[d:-d]) / (scale * scale - d)
        
        # 自适应滤波器
        elif method == 'local auto':
            # 自适应局部降低噪声滤波器
            sigma = 20 # 噪声的标准差，是一个需要另外估计的量
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    t = temp[i - r : i + r + 1, j - c : j + c + 1]
                    res[i, j] = temp[i, j] - (sigma ** 2) / np.var(t) * (temp[i, j] - np.mean(t))
        
        elif method == 'median auto':
            # 自适应中值滤波器
            def inner(i, j, t1, t2, t3, t4):
                try:
                    t = temp[t1 : t2, t3 : t4]
                    med_t, max_t, min_t = np.median(t), np.max(t), np.min(t)
                except:
                    return np.median(temp[t1 + 1 : t2 - 1, t3 + 1 : t4 - 1])
                a1, a2 = med_t - min_t, med_t - max_t
                if a1 > 0 and a2 < 0:
                    b1, b2 = temp[i, j] - min_t, temp[i, j] - max_t
                    if b1 > 0 and b2 < 0:
                        return temp[i, j]
                    else:
                        return med_t
                else:
                    if t2 - t1 < 9 and t4 - t3 < 9:
                        return inner(i, j, t1 - 1, t2 + 1, t3 - 1, t4 + 1)
                    else:
                        return med_t

            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = inner(i, j, i - r, i + r + 1, j - c, j + c + 1)

        return res[r : -r, c : -c]

if __name__ == '__main__':
    img1 = salt_noise(cv.imread('lena.jpg', 0), 500)
    img2 = cv.imread('lena.jpg', 0) + np.random.normal(0, 10, size = img1.shape)
    plt.figure(figsize = (12, 8))
    f = space_filter(img1)
    plt.subplot(231), plt.imshow(img1, cmap = 'gray'), plt.axis('off') # 加了椒盐噪声的图
    plt.subplot(232), plt.imshow(f.func_filter(3, 'local auto'), cmap = 'gray'), plt.axis('off') # 自适应局部降低噪声滤波器
    plt.subplot(233), plt.imshow(f.func_filter(3, 'median auto'), cmap = 'gray'), plt.axis('off') # 自适应中值滤波器
    f = space_filter(img2)
    plt.subplot(234), plt.imshow(img2, cmap = 'gray'), plt.axis('off') # 加了椒盐噪声的图
    plt.subplot(235), plt.imshow(f.func_filter(3, 'local auto'), cmap = 'gray'), plt.axis('off') # 自适应局部降低噪声滤波器
    plt.subplot(236), plt.imshow(f.func_filter(3, 'median auto'), cmap = 'gray'), plt.axis('off') # 自适应中值滤波器
    plt.show()