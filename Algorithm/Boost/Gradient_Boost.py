import numpy as np
from Criterion import loss
from Algorithm import util

from sklearn.datasets import load_wine
import time


class Gradient_Boost():
    def __init__(self, Loss='CrossEntropy', learning_rate=.01, terminal_iter=100, classifier='decision_tree', **kwargs):
        self.residual = loss.Set_Loss(Loss)
        self.classfier = classifier
        self.classfiers = list()
        self.learning_rate = learning_rate
        self.terminal = terminal_iter
        self.arg = kwargs

    def fit(self, x, y):
        model = util.Get_Classifer(self.classfier, **self.arg)
        model.fit(x, y)
        ypred = model.predict(x)
        self.classfiers.append(model)
        for i in range(self.terminal):
            y_train = self.residual.backward(y, ypred)
            print(ypred)
            print(y)
            print(y_train)
            print('----------------------------------------------------------------------')

            model = util.Get_Classifer(self.classfier, **self.arg)
            model.fit(x, y_train)
            ypred = ypred + self.learning_rate * model.predict(x)

            self.classfiers.append(model)

    def predict(self, x):
        ypred = self.classfiers[0].predict(x).astype(np.float64)
        for i in range(1, len(self.classfiers)):
            ypred += self.learning_rate * self.classfiers[i].predict(x)
        return np.rint(ypred)


if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)

    model = Gradient_Boost(max_depth=1)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.2f}ms'.format((end - start) * 1000))
    print(model.learning_rate)

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')
