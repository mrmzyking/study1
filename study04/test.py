import numpy as np

from ActivationFunction import *
from LossFunction import *
from GradientMethod import *
from dataset.mnist import load_mnist

class network(object):
    def __init__(self):
        self.paras = {}
        self.paras["W1"] = 0.01 * np.random.random((784, 500))
        self.paras["b1"] = np.zeros((500))
        self.paras["W2"] = 0.01 * np.random.random((500, 100))
        self.paras["b2"] = np.zeros((100))
        self.paras["W3"] = 0.01 * np.random.random((100, 10))
        self.paras["b3"] = np.zeros((10))

    def predict(self, x):
        W1, W2, W3 = self.paras["W1"], self.paras["W2"], self.paras["W3"]
        b1, b2, b3 = self.paras["b1"], self.paras["b2"], self.paras["b3"]
        h1 = np.dot(x, W1) + b1
        y1 = hyperbolictangentfunction(h1)
        h2 = np.dot(y1, W2) + b2
        y2 = softsignfunction(h2)
        h3 = np.dot(y2, W3) + b3
        y = softmax(h3)
        return y

    def loss(self, t, x):
        """
        计算损失率
        :param t: 监督数据
        :param x: 输入数据
        :return:返回损失了
        """
        y = self.predict(x)
        # print(y)
        lossval = crossentropyerror(t, y)
        return lossval

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(t, x)
        grads = {}
        grads['W1'] = numbergradient(loss_W, self.paras['W1'])
        grads['b1'] = numbergradient(loss_W, self.paras['b1'])
        grads['W2'] = numbergradient(loss_W, self.paras['W2'])
        grads['b2'] = numbergradient(loss_W, self.paras['b2'])
        grads['W3'] = numbergradient(loss_W, self.paras['W3'])
        grads['b3'] = numbergradient(loss_W, self.paras['b3'])
        return grads



if __name__ == '__main__':
    n = network()
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    iters_num = 100
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    batch_size = 10
    learning_rate = 0.01
    iter_per_epoch = max(train_size / batch_size, 1)
    for i in range(iters_num):
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 计算梯度
        grad = n.numerical_gradient(x_batch, t_batch)
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            n.params[key] -= learning_rate * grad[key]
