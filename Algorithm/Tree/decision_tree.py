import numpy as np

from Criterion import criterion as Criterion

import pandas as pd
from sklearn.datasets import load_iris
import time


class Node:
    def __init__(self):
        # links to the left and right child nodes
        self.right = None
        self.left = None

        # derived from splitting criteria
        self.column = None
        self.threshold = None

        # probability for object inside the Node to belong for each of the given classes
        self.prob = None
        # depth of the given node
        self.depth = None

        # if it is the root Node or not
        self.is_terminal = False


class decision_tree():
    def __init__(self, max_depth=3, min_samples_leaf=2, min_samples_split=2, verbose=False, criterion='gini'):
        self.criterion = Criterion.Criterion(criterion)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.verbose = verbose

        self.Tree = None

    def split(self, x, y):
        last = self.criterion(y, self.classes)
        # mean is always the best split. see linear regression
        mean = np.mean(x, axis=0).tolist()
        arr = x.T.tolist()
        bestcol = None
        bestthre = None
        bestGain = -999
        for i in range(len(mean)):
            a = np.array(arr[i])
            m = mean[i]
            tmp = (a > m)
            left, right = y[tmp == False], y[tmp]
            loss_left, loss_right = self.criterion(left, self.classes), self.criterion(right, self.classes)
            Gain = last - (loss_left * left.shape[0] / y.shape[0]) - (loss_right * right.shape[0] / y.shape[0])
            if Gain > bestGain:
                bestcol, bestthre, bestGain = i, m, Gain
        if bestGain == -999:
            return None, None, None, None, None, None
        x_col = x[:, bestcol]
        x_left, x_right = x[x_col <= bestthre, :], x[x_col > bestthre, :]
        y_left, y_right = y[x_col <= bestthre], y[x_col > bestthre]
        return bestcol, bestthre, x_left, x_right, y_left, y_right

    def buildtree(self, x, y, node):

        if node.depth >= self.max_depth:
            node.is_terminal = True
            return
        if x.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return
        if np.unique(y).shape[0] == 1:
            node.is_terminal = True
            return

        splitidx, thre, x_left, x_right, y_left, y_right = self.split(x, y)

        if splitidx is None:
            node.is_terminal = True
            return

        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return

        node.column = splitidx
        node.threshold = thre

        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.prob = Criterion.NodeProb(y_left, self.classes)

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.prob = Criterion.NodeProb(y_right, self.classes)

        self.buildtree(x_right, y_right, node.right)
        self.buildtree(x_left, y_left, node.left)

    def fit(self, x, y):
        if type(y) == np.ndarray:
            self.classes = list(np.unique(y))
        elif type(y) == pd.DataFrame:
            self.classes = list(np.unique(np.array(y)))
        elif type(y) == list:
            self.classes = list(set(y))
        else:
            raise TypeError("Unirecognize Data Type")

        if type(x) == pd.DataFrame:
            x = np.array(x)

        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.prob = Criterion.NodeProb(y, self.classes)
        self.buildtree(x, y, self.Tree)

    def predictSample(self, x, node):
        if node.is_terminal:
            return node.prob

        if x[node.column] > node.threshold:
            prob = self.predictSample(x, node.right)
        else:
            prob = self.predictSample(x, node.left)
        return prob

    def predict(self, X):
        if type(X) == pd.DataFrame:
            X = np.asarray(X)
        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.Tree))
            predictions.append(pred)
        return np.asarray(predictions)


if __name__ == '__main__':
    data = load_iris()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)
    print(X_train.shape)
    model = decision_tree(max_depth=10, min_samples_leaf=2, min_samples_split=2, criterion='gini')

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print((end - start) * 100)

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    print(y_pred)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=10)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print((end - start) * 100)
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')
