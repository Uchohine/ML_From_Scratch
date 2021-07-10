from Algorithm.SVM.SVC import svc
import numpy as np

import pandas as pd
from sklearn.datasets import load_wine
import time

class Node:
    def __init__(self):
        # links to the left and right child nodes
        self.classes = None
        self.classifier = None
        self.right = None
        self.left = None
        self.pred = None
        self.is_terminal = False


class svm():
    def __init__(self, lbda=.01, kernel='rbf', gamma='scale', tol=1e-3, C=1.0, **kargs):
        self.lbda = lbda
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.C = C
        self.arg = kargs
        self.Node = Node()

    def buildtree(self, c, x, y, node):
        if len(c) < 2:
            node.is_terminal = True
            node.pred = c[0]
            return
        idx = ((y == c[0]) | (y == c[-1]))
        sep = (float)((c[0] + c[-1]) / 2)
        xtrain = x[idx]
        ytrain = y[idx]
        for i in range(ytrain.shape[0]):
            ytrain[i] = 1 if ytrain[i] > sep else -1
        node.classifier = svc(self.lbda, self.kernel, self.gamma, self.tol, self.C, **self.arg).fit(xtrain, ytrain)
        node.left, node.right = Node(), Node()
        self.buildtree(c[0:-1], x, y, node.left)
        self.buildtree(c[1:], x, y, node.right)

    def fit(self, x, y):
        y = np.squeeze(y)
        candidates = np.arange(np.amax(y) + 1)
        self.buildtree(candidates, x, y, self.Node)

    def _predict(self, x, node):
        res = np.zeros(x.shape[0])
        if node.is_terminal:
            return node.pred
        pred = node.classifier.predict(x)
        neg = (pred < 0)
        pos = (pred > 0)
        if (neg == True).any():
            res[neg] = self._predict(x[neg], node.left)
        if (pos == True).any():
            res[pos] = self._predict(x[pos], node.right)
        return res.astype('int')

    def predict(self, x):
        return self._predict(x, self.Node)



if __name__ == '__main__':
    import scipy.io

    mat = scipy.io.loadmat('mnist.mat')


    X_train, X_val, y_train, y_val = mat['Xtr'].T.A, mat['Xte'].T.A, mat['ytr'], mat['yte']


    k = 'sigmoid'

    model = svm(kernel=k)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    print(y_pred)
    print(y_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = make_pipeline(StandardScaler(), SVC(kernel=k))
    start = time.time()
    model.fit(X_train, y_train.squeeze())
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))
    y_pred_ref = model.predict(X_val)
    print(y_pred_ref)
    print(f'Accuracy for sklearn SVC {accuracy_score(y_val, y_pred_ref)}')

