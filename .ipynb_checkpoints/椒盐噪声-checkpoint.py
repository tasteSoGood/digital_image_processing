#-*-coding: utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def salt_noise(pic, n = 100):
    def func(num):
        seed = np.random.randint(-1, 2)
        if seed == -1:
            return 0
        elif seed == 0:
            return num
        else:
            return 1
    inner = np.vectorize(func)
    res = pic.copy() / 255
    r, c = res.shape
    x, y = np.random.randint(0, r, n), np.random.randint(0, c, n)
    res[x, y] = inner(res[x, y])
    return res

def nonlinear_filter(f, shape = (3, 3), mode = 'median', expand = 'replicate', size = 'same'):
    temp = f.copy()
    temp = temp / 255 # 减小整数的舍入误差
    # 加边距
    r, c = np.array(shape) // 2
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
    if mode == 'median': # 中值滤波器
        for i in range(r, temp.shape[0] - r):
            for j in range(c, temp.shape[1] - c):
                res[i, j] = np.median(temp[i - r : i + r + 1, j - c : j + c + 1])
    elif mode == 'maximum': # 最大值
        for i in range(r, temp.shape[0] - r):
            for j in range(c, temp.shape[1] - c):
                res[i, j] = np.max(temp[i - r : i + r + 1, j - c : j + c + 1])
    elif mode == 'minimum': # 最小值
        for i in range(r, temp.shape[0] - r):
            for j in range(c, temp.shape[1] - c):
                res[i, j] = np.min(temp[i - r : i + r + 1, j - c : j + c + 1])

    if size == 'full':
        return res
    elif size == 'same':
        return res[r : -r, c : -c]

if __name__ == "__main__":
    img = cv.imread('lena.jpg', 0)
    noise = salt_noise(img)
    plt.subplot(121), plt.imshow(noise, cmap = 'gray')
    plt.subplot(122), plt.imshow(nonlinear_filter(noise), cmap = 'gray')
    plt.show()