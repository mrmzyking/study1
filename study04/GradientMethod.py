import numpy as np
"""
求数值的梯度
"""
def numbergradient(f,x):
    """
    中心差分法
    :param f: 计算的函数
    :param x: 输入数据集合
    :return: 返回梯度的集合
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad
