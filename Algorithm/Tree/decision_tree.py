import numpy as np
from Criterion import criterion as Criterion
from Criterion.loss import softmax
from Algorithm import util

import pandas as pd
from sklearn.datasets import load_wine
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
    def __init__(self, max_depth=3, min_samples_leaf=2, min_samples_split=2, verbose=False, eta = .08, criterion=None, type = 'classification'):
        if criterion != None:
            self.criterion = Criterion.Set_Criterion(criterion)
        else:
            if type == 'classification':
                self.criterion = Criterion.Set_Criterion('gini')
            else:
                self.criterion = Criterion.Set_Criterion('MAE')
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.eta = eta
        self.verbose = verbose
        self.type = type

        self.no_grad = False
        self.Tree = None

    def NodeProb(self, y):
        if self.type == 'classification':
            return np.sum(softmax(y), axis=0) / y.shape[0]
        else:
            return np.mean(y[:, -self.classes], axis = 0)

    def split(self, dataset):
        if self.type == 'classification':
            last = self.criterion(self.NodeProb(dataset[:, -self.classes:]))
        else:
            last = self.criterion(dataset[:,-self.classes])
        bestSplit = None
        bestThre = None
        bestGain = -999
        for i in range(dataset.shape[1] - self.classes):
            cur = dataset[:, i]
            if self.eta != 1:
                skip = (int)(1 // self.eta)
                idx = np.argsort(cur).tolist()
                can = list()
                j = 0
                while j < len(cur):
                    can.append(cur[idx[j]])
                    j += skip
            else:
                can = cur
            for val in can:
                dataset_right = dataset[cur > val]
                dataset_left = dataset[cur <= val]
                if dataset_left.shape[0] == 0 or dataset_right.shape[0] == 0:
                    continue
                if self.type == 'classification':
                    loss_right = self.criterion(self.NodeProb(dataset_right[:, -self.classes:]))
                    loss_left = self.criterion(self.NodeProb(dataset_left[:, -self.classes:]))
                    gain = last - (loss_left * dataset_left.shape[0] / dataset.shape[0]) - (
                            loss_right * dataset_right.shape[0] / dataset.shape[0])
                else:
                    gain = last - (self.criterion(dataset_left[:, -self.classes:]) * dataset_left.shape[
                        0] + self.criterion(dataset_right[:, -self.classes:]) *
                                   dataset_right.shape[0]) / dataset.shape[0]
                if gain > bestGain:
                    bestSplit = i
                    bestThre = val
                    bestGain = gain
        if bestGain == -999:
            return None, None, None, None

        idx = dataset[:, bestSplit]
        dataset_left, dataset_right = dataset[idx <= bestThre, :], dataset[idx > bestThre, :]
        return bestSplit,bestThre,dataset_left,dataset_right


    def buildtree(self, dataset, node):
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return
        if dataset.shape[0] < self.min_samples_split or dataset.shape[0] < (int)(1 // self.eta):
            node.is_terminal = True
            return
        if np.unique(np.argmax(dataset[:,-self.classes:], axis = 0)).shape[0] == 1 and self.type == 'classification':
            node.is_terminal = True
            return

        if np.unique(dataset[:,-self.classes:]).shape[0] == 1 and self.type != 'classification':
            node.is_terminal = True
            return

        splitidx, thre, dataset_left, dataset_right = self.split(dataset)

        if splitidx is None:
            node.is_terminal = True
            return

        if dataset_left.shape[0] < self.min_samples_leaf or dataset_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            return

        node.column = splitidx
        node.threshold = thre

        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.prob = self.NodeProb(dataset_left[:,-self.classes:])

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.prob = self.NodeProb(dataset_right[:, -self.classes:])

        self.buildtree(dataset_right, node.right)
        self.buildtree(dataset_left, node.left)

    def fit(self, x, y):
        if type(x) == pd.DataFrame:
            x = np.array(x)

        if self.type == 'classification':
            y, self.classes = util.to_one_hot(y)
            dataset = np.hstack((x, y))
        else:
            self.classes = 1
            dataset = np.hstack((x, np.split(y, y.shape[0])))

        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.prob = self.NodeProb(dataset[:,-self.classes:])
        self.buildtree(dataset, self.Tree)

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
            pred = self.predictSample(x, self.Tree)
            if self.no_grad:
                if self.type == 'classification':
                    pred = np.argmax(pred)
            predictions.append(pred)
        return np.asarray(predictions)


if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)
    model = decision_tree(max_depth=20, min_samples_leaf=2, min_samples_split=2)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))

    from sklearn.metrics import accuracy_score

    model.no_grad = True
    y_pred = model.predict(X_val)
    print(y_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=3)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')
