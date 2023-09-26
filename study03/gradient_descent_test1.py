import numpy as np
"""
求 Y = WX+B的梯度
X为一个（2，3）的矩阵
W为一个（3，2）的矩阵
B为一个（1，2）的矩阵

整体计算公式：y = wx + b
"""
x = np.array([[1.0,2.0,3.0],
          [4.0,5.0,6.0]])
w = np.array([[1.0,2.0],
              [2.0,3.0],
              [4.0,5.0]])
b = np.array(([1.0,2.0]))
def func_x(x):
    global w,b
    t1 =  np.dot(x,w)
    t2 = t1 + b
    return t2
def func_w(w):
    global x,b
    t1 = np.dot(x, w)
    t2 = t1 + b
    return t2
def fun_b(b):
    global x,w
    t1 = np.dot(x, w)
    t2 = t1 + b
    return t2
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad1 = (fxh1 - fxh2) / (2 * h)
        grad[idx] = grad1

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# y = func(x,w,b)
w1 = gradient_descent(func_w,w)
print(w1)


