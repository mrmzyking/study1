import numpy as np
"""
开发损失函数
"""
def meansquarederror(t,y):
    """
    均方差误差
    :param t: 监督数据
    :param y: 学习数据
    :return:
    """
    return 0.5 * np.sum((y - t) ** 2)

def crossentropyerror(t,y):
    """
    交叉熵误差
    :param t: 监督数据
    :param y: 学习数据
    :return: 
    """
    delta = 1e-7  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(t * np.log(y + delta))