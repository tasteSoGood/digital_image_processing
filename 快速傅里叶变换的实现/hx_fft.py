#-*-coding: utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

def conv(v1, v2):
    # 对两个向量进行卷积运算
    l1, l2, res = len(v1), len(v2), []
    v1, v2 = np.concatenate((np.zeros(l2 - 1), v1, np.zeros(l2 - 1))), v2[::-1]
    for i in range(l1 + l2 - 1):
        res.append(v1[i:i + l2].dot(v2.T))
    return res
    
def dft(v):
    # 离散傅里叶变换，这是一个复杂度为O(n^2)的算法
    N, vec = len(v), np.arange(len(v)).reshape(len(v), 1)
    mat = np.exp(-2 * np.pi / N * 1j) ** vec.dot(vec.T)
    return np.dot(v, mat)

def idft(v):
    # 离散傅里叶逆变换，这是一个复杂度为O(n^2)的算法
    N, vec = len(v), np.arange(len(v)).reshape(len(v), 1)
    mat = np.exp(2 * np.pi / N * 1j) ** vec.dot(vec.T)
    return (np.dot(v, mat) * 1 / N).real

def omega(N):
    v = np.arange(N).reshape(N, 1)
    mat = np.exp(-2 * np.pi / N * 1j) ** v.dot(v.T)
    return mat

if __name__ == "__main__":
    N = 20
    mat = np.round(omega(N).reshape(1, N ** 2), 5)
    l = list(set(mat[0]))
    plt.scatter(np.real(l), np.imag(l))
    plt.grid(True)
    plt.show()