#-*-coding: utf-8-*-
import numpy as np

def conv(v1, v2):
    # 对两个向量进行卷积运算
    l1, l2, res = len(v1), len(v2), []
    v1, v2 = np.concatenate((np.zeros(l2 - 1), v1, np.zeros(l2 - 1))), v2[::-1]
    for i in range(l1 + l2 - 1):
        res.append(v1[i:i + l2].dot(v2.T))
    return res
    
def dft(v):
    # 离散傅里叶变换，这是一个复杂度为O(n^2)的算法
    N, vec = len(v), np.arange(len(v)).reshape(1, len(v))
    mat = np.exp(-2 * np.pi / N * 1j) ** np.dot(vec.T, vec)
    return np.dot(v, mat)


def idft(v):
    # 离散傅里叶逆变换，这是一个复杂度为O(n^2)的算法
    N, vec = len(v), np.arange(len(v)).reshape(1, len(v))
    mat = np.exp(2 * np.pi / N * 1j) ** np.dot(vec.T, vec)
    return (np.dot(v, mat) * 1 / N).real

if __name__ == "__main__":
    a = np.array([1, 3, 4, 5])
    b = np.array([6, 0, 1, 2, 5])
    print(conv(a, b))

    a = np.append(a, np.zeros(4))
    b = np.append(b, np.zeros(3)) 
    print(np.round(idft(dft(a) * dft(b))))