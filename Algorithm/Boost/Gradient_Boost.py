import numpy as np
from Criterion import criterion as Criterion
from Algorithm import util

from sklearn.datasets import load_wine
import time


class Gradient_Boost():
    def __init__(self, Loss='CrossEntropy', learning_rate=.01, terminal_iter=100, classifier='decision_tree', **kwargs):
        self.residual = Criterion.Set_Loss(Loss)
        self.classfier = classifier
        self.classfiers = list()
        self.learning_rate = learning_rate
        self.terminal = terminal_iter
        self.arg = kwargs

    def fit(self, x, y):
        self.base = np.mean(y)
        ypred = np.ones(y.shape[0]) * self.base
        for i in range(self.terminal):

            y_train = list()
            for i in range(y.shape[0]):
                y_train.append(self.residual(ypred[i], y[i]))
            y_train = np.array(y_train)
            print(np.rint(ypred))
            print(y)
            print(y_train)

            model = util.Get_Classifer(self.classfier, **self.arg)
            model.fit(x, y_train)
            ypred = ypred + self.learning_rate * model.predict(x)

            self.classfiers.append(model)

    def predict(self, x):
        ypred = np.ones(x.shape[0]) * self.base
        print(ypred)
        for i in range(len(self.classfiers)):
            tmp = self.learning_rate * self.classfiers[i].predict(x)
            ypred += self.learning_rate * self.classfiers[i].predict(x)
        print(ypred)
        return ypred


if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)

    model = Gradient_Boost(max_depth=10)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.2f}ms'.format((end - start) * 1000))
    print(model.learning_rate)

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')
