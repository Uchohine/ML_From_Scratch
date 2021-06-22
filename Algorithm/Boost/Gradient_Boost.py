import numpy as np
from Criterion import loss
from Algorithm import util

from sklearn.datasets import load_wine
import time


class Gradient_Boost():
    def __init__(self, Loss='Deviance', learning_rate=1, terminal_iter=30, classifier='decision_tree', **kwargs):
        self.type = 0
        if Loss == 'Deviance':
            self.type = 1
        self.residual = loss.Set_Loss(Loss)
        self.classifier = classifier
        self.classifiers = list()
        self.learning_rate = learning_rate
        self.terminal = terminal_iter
        self.arg = kwargs

    def fit(self, x, y):
        if self.type:
            y,_ = util.to_one_hot(y)
        model = util.Get_Classifer(self.classifier, **self.arg)
        model.fit(x, y)
        ypred = model.predict(x)
        self.classifiers.append(model)
        for i in range(self.terminal):
            y_train = -self.residual.Gradient(y, ypred)
            model = util.Get_Classifer(self.classifier, **self.arg)
            model.fit(x, y_train)
            ypred += self.learning_rate * model.predict(x)

            self.classifiers.append(model)

    def predict(self, x):
        ypred = self.classifiers[0].predict(x)
        for i in range(1, len(self.classifiers)):
            ypred += self.learning_rate * self.classifiers[i].predict(x)
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
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.2f}ms'.format((end - start) * 1000))
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')


