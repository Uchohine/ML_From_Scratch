import numpy as np
from Criterion import loss
from Algorithm import util

from sklearn.datasets import load_wine
import time


class Gradient_Boost():
    def __init__(self, Loss='CrossEntropy', learning_rate=1, terminal_iter=10, classifier='decision_tree', **kwargs):
        self.residual = loss.Set_Loss(Loss)
        self.classfier = classifier
        self.classfiers = list()
        self.learning_rate = learning_rate
        self.terminal = terminal_iter
        self.arg = kwargs

    def fit(self, x, y):
        y,_ = util.to_one_hot(y)
        model = util.Get_Classifer(self.classfier, **self.arg)
        model.fit(x, y)
        ypred = model.predict(x)
        self.classfiers.append(model)
        for i in range(self.terminal):
            y_train = self.residual.backward(y, ypred)
            idx = np.where(np.argmax(y, axis = 1) - np.argmax(ypred, axis = 1) != 0)
            print(y[idx])
            print(ypred[idx])
            print(y_train[idx])
            model = util.Get_Classifer(self.classfier, **self.arg)
            model.fit(x, y_train)
            print((self.learning_rate * model.predict(x))[idx])
            print('-*-----------------------------------------------')
            ypred += self.learning_rate * model.predict(x)

            self.classfiers.append(model)

    def predict(self, x):
        ypred = self.classfiers[0].predict(x)
        for i in range(1, len(self.classfiers)):
            ypred += self.learning_rate * self.classfiers[i].predict(x)
        return np.argmax(ypred, axis = 1)


if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)

    model = Gradient_Boost(max_depth=3)

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
    print('elapsed time : {:.5f}s'.format((end - start)))
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')


