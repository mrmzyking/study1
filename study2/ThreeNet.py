import numpy as np
import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


class FourNetWork(object):
    def __init__(self):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(784, 500)
        self.params['b1'] = np.zeros(500)
        self.params['W2'] = 0.01 * np.random.randn(500, 200)
        self.params['b2'] = np.zeros(200)
        self.params['W3'] = 0.01 * np.random.randn(200, 50)
        self.params['b3'] = np.zeros(50)
        self.params['W4'] = 0.01 * np.random.randn(50, 10)
        self.params['b4'] = np.zeros(10)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def predict(self, x):
        W1, W2, W3, W4 = self.params['W1'], self.params['W2'], self.params['W3'], self.params['W4']
        b1, b2, b3, b4 = self.params['b1'], self.params['b2'], self.params['b3'], self.params['b4']

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = self.sigmoid(a3)
        a4 = np.dot(z3, W4) + b4
        y = self.softmax(a4)
        return y

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def numerical_gradient(self, f, x):
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
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val  # 値を元に戻す
            it.iternext()
        return grad

    def loss(self, x, t):
        y = self.predict(x)
        lossval = self.cross_entropy_error(y, t)
        return lossval
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient_end(self, x, t):
        loss_W = lambda w: self.loss(x, t)
        grads = {}
        grads['W1'] = self.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self.numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = self.numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = self.numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = self.numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = self.numerical_gradient(loss_W, self.params['b4'])
        return grads


if __name__ == '__main__':
    network = FourNetWork()
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 平均每个epoch的重复次数
    # 超参数
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
        grad = network.numerical_gradient_end(x_batch, t_batch)
        # grad = network.gradient(x_batch, t_batch) # 高速版!
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        # print("{} loss = {} ".format(datetime.now(),loss))
        train_loss_list.append(loss)
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
