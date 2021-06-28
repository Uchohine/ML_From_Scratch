import numpy as np
import numexpr as ne
from scipy.linalg.blas import sgemm
import qpsolvers

import pandas as pd
from sklearn.datasets import load_iris
import time


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


def kernel_sigmoid(x, y, coef=0.0, gamma=.5):
    return np.tanh(gamma * np.matmul(x, y.T) + coef)


def get_kernel(name='linear'):
    switcher = {
        'linear': kernel_linear,
        'poly': kernel_poly,
        'RBF': kernel_RBF,
        'sigmoid': kernel_sigmoid
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Kernel Named: "{name}".'.format(name=name));
    return res


def to_one_hot(x):
    res = np.zeros((x.size, x.max() + 1))
    res[np.arange(x.size), x] = 1
    return res


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class svm():
    def __init__(self, lbda=.01, kernel='linear', gamma='scale', tol=1e-5, C=1.0, **kargs):
        self.weight = None
        self.lbda = lbda
        self.kernel = get_kernel(kernel)
        self.gamma = gamma
        self.tol = tol
        self.C = C
        self.arg = kargs


    def fit(self, x, y):
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1
        self.mean = np.mean(x, axis=0)
        x = (x - self.mean) / self.std
        if self.gamma == 'auto':
            self.arg['gamma'] = 1 / x.shape[1]
        else:
            self.arg['gamma'] = 1 / (x.shape[1] * np.var(x))
        m = x.shape[0]
        K = self.kernel(x, x, **self.arg)
        H = (np.matmul(y, y.T) * K).astype('float64')
        f = -np.ones(m)
        G = np.vstack((np.eye(m) * -1, np.eye(m)))
        h = np.hstack((np.zeros(m), np.ones(m) * self.C))
        Aeq = y.reshape(1, -1).astype('float64')
        b = np.zeros(1)
        lb = np.zeros(m)
        ub = np.ones(m) * self.C
        a = qpsolvers.solve_qp(H, f, G=G, h=h, A=Aeq, b=b, lb=lb, ub=ub, solver='cvxopt', feastol=self.tol)
        idx = a > self.tol
        ind = np.arange(len(a))[idx]
        self.a = a[idx]
        self.y = np.squeeze(y)[idx]
        self.x = x[idx]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.y[n]
            self.b -= np.sum(self.a * self.y * K[ind[n], idx])
        self.b /= len(self.a)

    def project(self, x):
        x = (x - self.mean) / self.std
        k = self.kernel(self.x, x, **self.arg)
        ypred = np.sum(np.matmul(np.diag(self.a * self.y), k), axis=0)
        return ypred + self.b

    def predict(self, x):
        return np.sign(self.project(x))


if __name__ == '__main__':
    import scipy.io
    mat = scipy.io.loadmat('problem_4_2_a.mat')
    X_val, X_train, y_val, y_train = mat['Xtest'], mat['Xtrain'], mat['Ytest'], mat['Ytrain']
    model = svm(kernel='sigmoid')

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))

    from sklearn.metrics import accuracy_score

    y_pred_self = model.predict(X_val)
    print(y_pred_self)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred_self)}')

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    start = time.time()
    model.fit(X_train, np.squeeze(y_train))
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))
    y_pred_ref = model.predict(X_val)
    print(y_pred_ref)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred_ref)}')
    print(np.argwhere(y_pred_ref != y_pred_self))

