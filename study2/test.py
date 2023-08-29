# coding: utf-8
import sys, os
from datetime import datetime

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def sigmoid(x: np):
    return 1 / (1 + np.exp(-x))


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]


def cross_entropy_error1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
    # y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
train_size = x_train.shape[0]
network = init_network()
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
y = predict(network, x_batch)
loss = cross_entropy_error(y, t_batch)
print("predict = {} \r\n t_batch = {} \r\n loss = {}".format(y, t_batch, loss))
# x, t = get_data()
# network = init_network()
# batch_size = 5  # 批数量
# accuracy_cnt = 0
# for i in range(0, len(x), batch_size):
#     x_batch = x[i:i + batch_size]
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#     print("{} info : y = {} , p = {}".format(datetime.now(),y_batch,p))
#     accuracy_cnt += np.sum(p == t[i:i + batch_size])
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
