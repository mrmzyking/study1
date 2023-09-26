import numpy as np

"""
开发激活函数
"""


def identityfunction(x):
    """
    恒等式激活函数，输入即输出
    :param x:
    :return:
    """
    return x


def stepfunction(x):
    """
    阶跃函数
    f(x) =
          1: x > 0
          0: x <= 0
    :param x:
    :return:
    """
    y = (x > 0).astype(int)
    return y


def logicalfunction(x):
    """
    逻辑函数
    f(x) = 1 / (1 + e ^ (-x))
    :param x:
    :return:
    """
    y = 1 / (1 + np.exp(-x))
    return y


def hyperbolictangentfunction(x):
    """
    双曲正切函数
    f(x) = (e^x - e^-x)/(e^x + e^-x)
    :param x:
    :return:
    """
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y


def arctangentfunction(x):
    """
    反正切函数
    f(x) = tan-1(x)
    :param x:
    :return:
    """
    y = np.tan(x)
    return y


def softsignfunction(x):
    """
    Softsign函数
    f(x) = x / (1 +|x|)
    :param x:
    :return:
    """
    return x / (1 + np.abs(x))


def ISRUfunction(x):
    """
    反平方根函数
    f(x) = x / (/(1+ax))
    :param x:
    :return:
    """
    return x / numpy.square(1 + 0.01 * x)


def ReLUfunction(x):
    """
    线性整流函数(ReLU)
    f(x) = :
           0 x < 0
           x x >= 0
    :param x:
    :return:
    """
    return np.maximum(0, x)


def LeakyReLUfunction(x):
    """
    带泄露线性整流函数(Leaky ReLU)
    f(x) = :
           0.01x x < 0
           x x   >= 0
    :param x:
    :return:
    """
    return np.maximum(0.01 * x, x)


def PReLUfunction(x):
    """
    参数化线性整流函数(PReLU)
    f(x) = :
           0.5x x < 0
           x x   >= 0
    :param x: 
    :return: 
    """
    return np.maximum(0.5 * x, x)


def RReLUfunction(a, x):
    """
    带泄露随机线性整流函数(RReLU)
    f(x) = :
           a*x x < 0
           x x   >= 0
    :param x:
    :return:
    """
    return np.maximum(a * x, x)


def ELUfunction(x):
    """
    指数线性函数(ELU)
    f(x) = :
           a*(e^x - 1) x < 0
           x x   >= 0
    :param x:
    :return:
    """
    return np.maximum(0.5 * (np.exp(x) - 1), x)


def softmax(x):
    """
    softmax 函数
    f(x) = 
    :param x: 
    :return: 
    """
    c = np.max(x)
    exp_a = np.exp(x - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y



