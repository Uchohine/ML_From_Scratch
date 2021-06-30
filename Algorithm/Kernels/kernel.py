import numpy as np
import numexpr as ne
from scipy.linalg.blas import sgemm

def kernel_poly(x, z, degree=3, gamma=.1, coef0=0.0):
    return np.power(gamma * np.matmul(x, z.T) + coef0, degree)


def kernel_RBF(X, Y, gamma=.1, var=1):
    X_norm = -gamma * np.einsum('ij,ij->i', X, X)
    Y_norm = -gamma * np.einsum('ij,ij->i', Y, Y)
    return ne.evaluate('v * exp(A + B + C)', { \
        'A': X_norm[:, None], \
        'B': Y_norm[None, :], \
        'C': sgemm(alpha=2.0 * gamma, a=X, b=Y, trans_b=True), \
        'g': gamma, \
        'v': var \
        })


def kernel_linear(x, z, **kwargs):
    return np.matmul(x, z.T)


def kernel_sigmoid(x, y, coef0=0.0, gamma=.5):
    return np.tanh(gamma * np.matmul(x, y.T) + coef0)


def get_kernel(name='linear'):
    switcher = {
        'linear': kernel_linear,
        'poly': kernel_poly,
        'rbf': kernel_RBF,
        'sigmoid': kernel_sigmoid
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Kernel Named: "{name}".'.format(name=name));
    return res