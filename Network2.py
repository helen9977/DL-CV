"""
在network的基础上
增加了交叉熵损失函数，
使用了正则化降低过拟合，
同时还优化了初始权重的设定方法

"""

import numpy as np
import random
import sys
import json

from sklearn.externals._arff import xrange

'''
 二次cost函数
'''


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y)**2  # 矩阵求范数
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


'''
 交叉熵cost函数
'''


class CrossEntropyCost(object):
    @staticmethod
    # 为了数值稳定性，有可能出现（y=1，a=1）的情况，0*log(0)=nan
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))


    @staticmethod
    # 保持接口一致性
    def delta(z, a, y):
        return a-y


class Network (object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weights_initializer()
        self.cost = cost

    def default_weights_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]  #  标准正态分布


        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]  # 均值为0 标准差为1/sqrt（x）

    def large_weights_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    '''
    eta : 学习率
    lamda ：正则化参数
    evaluation_data : 验证集或测试集
    monitor ： 返回每一个epoch的cost和accuracy
    '''
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lamda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):

        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lamda, len(training_data))

            print('Epoch %s training complete' % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lamda)
                training_cost.append(cost)
                print('Cost on training data : {}'.format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print('accuracy on training data : {} / {}'.format(accuracy, n))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lamda, convert=True)
                evaluation_cost.append(cost)
                print('Cost on evaluation data: {}'.format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print('Accuracy on evaluation data: {} / {}'.format(self.accuracy(evaluation_data), n_data))

            print()

        return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy


    '''
    利用BP算法更新网络的权重和偏置
    lamda 正则化
    '''

    def update_mini_batch(self, mini_batch, eta, lamda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)  # 偏导数
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 求和
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta * (lamda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        # delta = self.cost(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 倒数第二层 开始反向更新
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    # training data -- true ,  validation,test data -- false
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    # training data -- false , validation,test data -- true
    def total_cost(self, data, lamda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)

        cost += 0.5 * (lamda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {
            'sizes': self.sizes,
            'weight': [w.tolist()for w in self.weights],
            'bias': [b.tolist()for b in self.biases],
            'cost': str(self.cost.__name__)
        }
        with open(filename, 'w') as f:
            json.dump(data, f)


def load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


# 生成10*1的矩阵 对应标签为1
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# 求导
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
