import numpy as np
from matplotlib import pyplot as plt


class GateCircl(object):
    def __init__(self):
        self.weight1: float = 0.0
        self.weight2: float = 0.0
        self.thera: float = 0.0

    def set(self, w1: float, w2: float, thera: float):
        self.weight1 = w1
        self.weight2 = w2
        self.thera = thera

    def ANDgate(self, x1, x2):
        tmp = x1 * self.weight1 + x2 * self.weight2
        if tmp < self.thera:
            return 0
        else:
            return 1

    def step_function(self,x:np):
        y = x > 0
        return y.astype(dtype=np.int64)

    def sigmoid(self,x:np):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    g = GateCircl()
    x = np.arange(-10.0, 10.0, 0.1)
    y = g.sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    plt.show()
