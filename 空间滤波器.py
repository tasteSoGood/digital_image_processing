# -*-coding: utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from functools import reduce

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
        if method == 'arithmetic':
            # 算术均值
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.mean(temp[i - r : i + r + 1, j - c : j + c + 1])
        elif method == 'geometric':
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = reduce(lambda a, b: a * b, temp[i - r : i + r + 1, j - c : j + c + 1].ravel())
            res = res ** (1 / (scale ** 2))
        elif method == 'harmonic':
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = scale * scale / np.sum(1 / temp[i - r : i + r + 1, j - c : j + c + 1])
        elif method == 'inv harmonic':
            q = 2
            for i in range(r, temp.shape[1] - c):
                for j in range(c, temp.shape[1] - c):
                    res[i, j] = np.sum(temp[i - r : i + r + 1, j - c : j + c + 1] ** (q + 1)) / np.sum(temp[i - r : i + r + 1, j - c : j + c + 1] ** q)

        return res[r : -r, c : -c]

if __name__ == '__main__':
    img = cv.imread('lena.jpg', 0)
    img = img + np.random.normal(0, 20, size = img.shape)
    f = space_filter(img)
    plt.figure(figsize = (12, 8))
    plt.subplot(231), plt.imshow(img, cmap = 'gray'), plt.axis('off') # 加噪的图像
    plt.subplot(232), plt.imshow(f.func_filter(3, 'arithmetic'), cmap = 'gray'), plt.axis('off') # 算术均值滤波器
    plt.subplot(233), plt.imshow(f.func_filter(3, 'geometric'), cmap = 'gray'), plt.axis('off') # 几何均值滤波器
    plt.subplot(234), plt.imshow(f.func_filter(3, 'harmonic'), cmap = 'gray'), plt.axis('off') # 谐波均值滤波器
    plt.subplot(235), plt.imshow(f.func_filter(3, 'inv harmonic'), cmap = 'gray'), plt.axis('off') # 逆谐波均值滤波器
    plt.show()