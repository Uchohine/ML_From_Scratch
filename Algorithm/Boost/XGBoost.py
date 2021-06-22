import numpy as np
from Criterion import loss
from Algorithm import util
from Algorithm.Tree.decision_tree import decision_tree
from Algorithm.Tree.XG_tree import XGTree

from sklearn.datasets import load_wine
import time


class XGBoost():
    def __init__(self, Loss='CrossEntropy', learning_rate=.1, terminal_iter=50, **kwargs):
        self.type = 0
        if Loss == 'CrossEntropy':
            self.type = 1
        self.loss = loss.Set_Loss(Loss)
        self.learning_rate = learning_rate
        self.terminal_iter = terminal_iter
        self.classifiers = list()
        self.arg = kwargs

    def fit(self, x, y):
        if self.type:
            y, _ = util.to_one_hot(y)
            model = decision_tree()
        else:
            model = decision_tree(type='regression')
        model.fit(x, y)
        ypred = model.predict(x)
        self.classifiers.append(model)
        g = self.loss.Gradient(y,ypred)
        h = self.loss.Hessian(y,ypred)

        for i in range(self.terminal_iter):
            model = XGTree(**self.arg)
            T = model.fit(x, g, h)
            ypred -= self.learning_rate * model.predict(x)
            g = self.loss.Gradient(y, ypred)
            h = self.loss.Hessian(y, ypred)

            self.classifiers.append(model)
            self.arg['T'] = T

    def predict(self, x):
        ypred = self.classifiers[0].predict(x)
        for i in range(1, len(self.classifiers)):
            ypred -= self.learning_rate * self.classifiers[i].predict(x)
        if self.type:
            return np.argmax(ypred, axis=1)
        return ypred


if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)

    model = XGBoost(max_depth=3, lbda = 0.01, eta = 0.08)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.2f}ms'.format((end - start) * 1000))

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    print(y_pred)
    print(y_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.2f}ms'.format((end - start) * 1000))
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')
            



