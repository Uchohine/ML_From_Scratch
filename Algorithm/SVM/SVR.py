from Algorithm.Kernels.kernel import get_kernel
import numpy as np
import qpsolvers

import pandas as pd
from sklearn.datasets import load_iris
import scipy.io as sio
import time


class svr():
    def __init__(self, lbda=.01, kernel='linear', gamma='scale', epsilon=.1, C=1.0, **kargs):
        self.weight = None
        self.lbda = lbda
        self.kernel = get_kernel(kernel)
        self.gamma = gamma
        self.eps = epsilon
        self.C = C
        self.arg = kargs


    def fit(self, x, y):
        if self.gamma == 'auto':
            self.arg['gamma'] = 1 / x.shape[1]
        elif self.gamma == 'scale':
            self.arg['gamma'] = 1 / (x.shape[1] * np.var(x))
        y = np.squeeze(y)
        m = x.shape[0]
        Aeq = np.array(m * [1, -1]).reshape((1, -1)).astype('float64')
        ub = np.array(m * [self.C, self.C])
        lb = np.zeros((2*m))
        f = list()
        for i in range(m):
            f.append(-self.eps + y[i])
            f.append(-self.eps - y[i])
        f = -np.array(f)
        H = np.zeros((2*m, 2*m))
        K = self.kernel(x, x, **self.arg)
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                H[2 * i, 2 * j], H[2 * i + 1, 2 * j + 1] = K[i, j], K[i, j]
                H[2 * i, 2 * j + 1], H[2 * i + 1, 2 * j] = -K[i, j], -K[i, j]
        b = np.zeros(1)
        a = qpsolvers.solve_qp(H.astype('float64'), f, G=None, h=None, A=Aeq,\
                               b=b, lb=lb, ub=ub, solver='cvxopt')
        tmup = list()
        tmdp = list()
        for i in range(a.shape[0]):
            if i % 2 == 0:
                tmup.append(a[i])
            else:
                tmdp.append(a[i])
        self.support_ = list()
        for i in range(x.shape[0]):
            self.support_.append(i)
        self.support_ = np.array(self.support_)
        self.x = x
        self.a = np.array(tmup) - np.array(tmdp)
        self.b = np.mean(y - np.matmul(self.a, K))
        return self




    def predict(self, x):
        k = self.kernel(self.x, x, **self.arg)
        ypred = np.sum(np.matmul(np.diag(self.a), k), axis=0)
        return ypred + self.b


if __name__ == '__main__':
    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt

    # #############################################################################
    # Generate sample data
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # #############################################################################
    # Add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    svr_rbf = svr(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                   coef0=1)
    svr_poly = svr(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                   coef0=1)

    # #############################################################################
    # Look at the results
    lw = 2

    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()

