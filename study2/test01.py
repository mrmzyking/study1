import numpy as np
import matplotlib.pylab as plt


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_2(x):
    if len(x) == 3:
        return (4 * x[0] - 2 * x[2] - 1.5 * x[1]) ** 5 + (6 * x[1] + 5 * x[0] - 3 * x[2]) ** 4 + (
                    3.5 * x[2] - 2.5 * x[1] - 5.5 * x[0]) ** 3
    elif len(x) == 2:
        return np.sum(x ** 2)
    else:
        return np.sum(x * 2 + 3)


def numerical_gradient(f, x):
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


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        print(" {} current grad = {} , x = {}".format(i, grad, x))
    return x


if __name__ == '__main__':
    init_x = np.array([10.0, 20.0, 30.0])
    x = gradient_descent(function_2, init_x=init_x, lr=0.01, step_num=1000)
    print(x)
