import numpy as np
import numexpr as ne
from scipy.linalg.blas import sgemm
import qpsolvers

import pandas as pd
from sklearn.datasets import load_iris
import time


def kernel_poly(x, z, degree = 2, coef0 = 1):
    return np.power(np.matmul(x, z.T) + coef0, degree)

def kernel_RBF(X,Y, gamma = .1, var = 1):
    X_norm = -gamma*np.einsum('ij,ij->i',X,X)
    Y_norm = -gamma * np.einsum('ij,ij->i', Y, Y)
    return ne.evaluate('v * exp(A + B + C)', {\
        'A' : X_norm[:,None],\
        'B' : Y_norm[None,:],\
        'C' : sgemm(alpha=2.0*gamma, a=X, b=Y, trans_b=True),\
        'g' : gamma,\
        'v' : var\
    })

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
    def __init__(self, lbda = .01, kernel = 'linear', **kargs):
        self.weight = None
        self.lbda = lbda
        self.kernel = get_kernel(kernel)
        self.arg = kargs

    def fit(self, x, y):
        H = np.matmul(y, y.T) * self.kernel(x,x,**self.arg) / (4 * self.lbda)
        f = -np.ones(x.shape[0])
        Aeq = y.T.astype('float64')
        b = np.zeros(1)
        alpha = qpsolvers.solve_qp(H,f,None,None,Aeq,b,sym_proj=True)
        self.alpha = alpha * np.squeeze(y)
        self.x = x

    def predict(self,x):
        k = self.kernel(self.x, x, **self.arg)
        ypred = np.sum(np.matmul(np.diag(self.alpha), k), axis=0) / (2*self.lbda)
        return np.sign(ypred)


if __name__ == '__main__':
    import scipy.io
    mat = scipy.io.loadmat('problem_4_2_a.mat')
    X_val, X_train, y_val, y_train = mat['Xtest'], mat['Xtrain'], mat['Ytest'], mat['Ytrain']
    model = svm(kernel='linear')

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))

    from sklearn.metrics import accuracy_score

    model.no_grad = True
    y_pred = model.predict(X_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto'))
    start = time.time()
    model.fit(X_train, np.squeeze(y_train))
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')















