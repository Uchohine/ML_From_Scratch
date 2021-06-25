import numpy as np
import qpsolvers

import pandas as pd
from sklearn.datasets import load_iris
import time


def kernel_poly(x, z, degree, intercept):
    return np.power(np.matmul(x, z.T) + intercept, degree)

def kernel_RBF(x, z, sigma):
    n = x.shape[0]
    m = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))

def kernel_linear(x, z):
    return np.matmul(x, z.T)

def kernel_sigmoid(x, y, coef = 1, gamma = .5):
    return np.tanh(gamma * np.matmul(x, y.T) + coef)

def get_kernel(name = 'linear'):
    switcher = {
        'linear': kernel_linear,
        'poly':kernel_poly,
        'RBF':kernel_RBF,
        'sigmoid':kernel_sigmoid
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Kernel Named: "{name}".'.format(name=name));
    return res

def to_one_hot(x):
    res = np.zeros((x.size, x.max()+1))
    res[np.arange(x.size), x] = 1
    return res

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
class svm():
    def __init__(self, lbda = .1, kernel = 'linear', **kargs):
        self.weight = None
        self.lbda = lbda
        self.kernel = get_kernel(kernel)
        self.arg = kargs

    def fit(self, x, y):
        print(x.shape)
        H = np.matmul(y, y.T) * np.matmul(x, x.T) / (4 * self.lbda)
        print(np.linalg.eigh(H))
        f = -np.ones(x.shape[0])
        Aeq = y.T.astype('float64')
        b = np.zeros(1)
        print(Aeq.shape)
        alpha = qpsolvers.solve_qp(H,f,None,None,Aeq,b)


if __name__ == '__main__':
    import scipy.io
    mat = scipy.io.loadmat('problem_4_2_a.mat')
    Xtest, Xtrain, Ytest, Ytrain = mat['Xtest'], mat['Xtrain'], mat['Ytest'], mat['Ytrain']
    model = svm()
    model.fit(Xtrain, Ytrain)











