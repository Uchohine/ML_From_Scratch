import numpy as np
from Criterion import loss
from Algorithm import util
from Algorithm.Tree.decision_tree import Node
from Algorithm.Tree.decision_tree import decision_tree

from sklearn.datasets import load_wine
import time

class XGTree():
    def __init__(self, max_depth=3, min_samples_leaf=2, min_samples_split=2, lbda=0, gamma=0, T=0,  use_appro=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.lbda = lbda
        self.use_appro = use_appro
        self.gamma = gamma
        self.T = T
        self.leaves = 0

        self.tree = None

    def NodeProb(self, g, h):
        return np.sum(g, axis=0) / (np.sum(h, axis=0) + self.lbda + np.finfo(float).eps)

    def Score(self, g, h):
        return 0.5 * (np.sum(g) ** 2 / (np.sum(h) + self.lbda + np.finfo(float).eps)) + self.gamma * self.T

    def split(self, x, g, h):
        last = self.Score(g, h)
        bestSplit = None
        bestThre = None
        bestGain = -999
        if self.use_appro:
            candidate = list()
        else:
            candidate = x
        for i in range(candidate.shape[1]):
            cur = x[:,i]
            can = candidate[:,i]
            for val in can:
                left = (cur <= val)
                right = (cur > val)
                score = self.Score(g[left], h[left]) + self.Score(g[right], h[right]) - last
                if score > bestGain:
                    bestSplit = i
                    bestThre = val
                    bestGain = score
        if bestGain == -999:
            return None, None, None, None, None, None, None, None

        idx = x[:, bestSplit]
        x_left, x_right = x[idx <= bestThre, :], x[idx > bestThre, :]
        g_left, g_right = g[idx <= bestThre, :], g[idx > bestThre, :]
        h_left, h_right = h[idx <= bestThre, :], h[idx > bestThre, :]

        return bestSplit, bestThre, x_left, x_right, g_left, g_right, h_left, h_right

    def buildtree(self, x, g, h, node):
        self.leaves += 1
        
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if x.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        bestSplit, bestThre, x_left, x_right, g_left, g_right, h_left, h_right = self.split(x,g,h)

        node.column = bestSplit
        node.threshold = bestThre

        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.prob = self.NodeProb(g_left, h_left)

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.prob = self.NodeProb(g_right, h_right)

        self.buildtree(x_left, g_left, h_left, node.left)
        self.buildtree(x_right, g_right, h_right, node.right)

    def fit(self, x, g, h):
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.prob = self.NodeProb(g,h)
        self.buildtree(x, g, h, self.Tree)
        return self.leaves

    def predictSample(self, x, node):
        if node.is_terminal:
            return node.prob

        if x[node.column] > node.threshold:
            prob = self.predictSample(x, node.right)
        else:
            prob = self.predictSample(x, node.left)
        return prob

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.predictSample(x, self.Tree)
            predictions.append(pred)
        return np.asarray(predictions)

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
            idx = np.where(np.argmax(y,axis=1) - np.argmax(ypred,axis=1) != 0)
            print(idx)
            print(y[127,:])
            print(ypred[127,:])
            print(g[127,:])
            print(h[127,:])
            print('---------------------------------------------------')
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

    model = XGBoost(max_depth=3)

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
            



