# -*-coding: utf-8-*-
## 实现了一个简单的线性空间滤波器（灰度图像）
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def imfilter(f, w, mode = 'conv', expand = 'replicate', size = 'same'):
    temp = f.copy()
    temp = temp / 255 # 减小整数的舍入误差
    # 加边距
    r, c = np.array(w.shape) // 2
    if expand == 'zero':
        temp = cv.copyMakeBorder(temp, r, r, c, c, cv.BORDER_CONSTANT, value = [0])
    elif expand == 'replicate':
        temp = cv.copyMakeBorder(temp, r, r, c, c, cv.BORDER_REPLICATE)
    elif expand == 'symmetric':
        temp = cv.copyMakeBorder(temp, r, r, c, c, cv.BORDER_REFLECT)
    else:
        temp = cv.copyMakeBorder(temp, r, r, c, c, cv.BORDER_CONSTANT, value = [0])
    # 运算
    res = temp.copy()
    if mode == 'conv':
        w = w[::-1, ::-1] # 卷积
    elif mode == 'corr':
        w
    for i in range(r, temp.shape[0] - r):
        for j in range(c, temp.shape[1] - c):
            res[i, j] = np.dot(temp[i - r : i + r + 1, j - c : j + c + 1].ravel(), w.ravel().T)
    if size == 'full':
        return res
    elif size == 'same':
        return res[r : -r, c : -c]

if __name__ == '__main__':
    img = cv.imread('lena.jpg', 0)
    plt.subplot(221), plt.imshow(img, cmap = 'gray'), plt.title('original')
    plt.subplot(222)
    plt.imshow(imfilter(img, np.ones((3, 3)), size = 'same'), cmap = 'gray')
    plt.title('filterized_3')
    plt.subplot(223)
    plt.imshow(imfilter(img, np.ones((5, 5)), size = 'same'), cmap = 'gray')
    plt.title('filterized_5')
    plt.subplot(224)
    plt.imshow(imfilter(img, np.ones((7, 7)), expand = 'replicate', size = 'same'), cmap = 'gray')
    plt.title('filterized_7')
    plt.show()