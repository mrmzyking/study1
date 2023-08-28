import numpy as np


class ThreeNetWork(object):
    def __init__(self):
        self.W1 = np.zeros(shape=(2, 3), dtype=float)
        self.b1 = np.zeros(shape=(1, 3), dtype=float)
        self.W2 = np.zeros(shape=(3, 2), dtype=float)
        self.b2 = np.zeros(shape=(1, 2), dtype=float)
        self.W3 = np.zeros(shape=(2, 1), dtype=float)
        self.b3 = np.zeros(shape=(1, 1), dtype=float)

    def setW1(self, w1: np):
        if w1.shape[0] == 2 and w1.shape[1] == 3:
            self.W1 = w1

    def setW2(self, w2: np):
        if w2.shape[0] == 3 and w2.shape[1] == 2:
            self.W2 = w2

    def setW3(self, w3: np):
        if w3.shape[0] == 2 and w3.shape[1] == 2:
            self.W3 = w3

    def setB1(self, b1: np):
        if b1.shape[0] == 1 and b1.shape[1] == 3:
            self.b1 = b1

    def setB2(self, b2: np):
        if b2.shape[0] == 1 and b2.shape[1] == 2:
            self.b2 = b2

    def setB3(self, b3: np):
        if b3.shape[0] == 1 and b3.shape[1] == 2:
            self.b3 = b3

    def init_network(self):
        network = {}
        network['W1'] = self.W1
        network['b1'] = self.b1
        network['W2'] = self.W2
        network['b2'] = self.b2
        network['W3'] = self.W3
        network['b3'] = self.b3
        return network

    def sigmoid(self, x: np):
        return 1 / (1 + np.exp(-x))

    def identity_function(self, x):
        return x

    def softmax(self,a):
        c = np.max(a)
        exp_a = np.exp(a - c)  # 溢出对策
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y
    def forword(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.identity_function(a3)
        return y


if __name__ == '__main__':
    t = ThreeNetWork()
    t.setW1(np.array([[0.2, 0.8, 0.6],[0.5, 0.7, 0.9]]))
    t.setB1(np.array([[0.4, 0.5, 0.6]]))
    t.setW2(np.array([[0.5, 0.4], [0.8, 0.7], [0.6, 0.3]]))
    t.setB2(np.array([[0.2, 0.7]]))
    t.setW3(np.array([[0.5,0.6], [0.5,0.6]]))
    t.setB3(np.array([0.5,0.9]))
    network = t.init_network()
    y = t.forword(network, np.array([0.5, 0.6]))
    print(y)
