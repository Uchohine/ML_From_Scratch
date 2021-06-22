import numpy as np
from Algorithm.Tree.decision_tree import Node

class XGTree():
    def __init__(self, max_depth=3, min_samples_leaf=2, min_samples_split=2, lbda=0, gamma=0, T=0, eta=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.lbda = lbda
        self.gamma = gamma
        self.eta = eta
        self.T = T
        self.leaves = 0

        self.tree = None

    def NodeProb(self, g, h):
        return np.sum(g, axis=0) / (np.sum(h, axis=0) + self.lbda + np.finfo(float).eps)

    def Score(self, g, h):
        return 0.5 * (np.sum(g) ** 2 / (np.sum(h) + self.lbda + np.finfo(float).eps))

    def split(self, x, g, h):
        last = self.Score(g, h)
        bestSplit = None
        bestThre = None
        bestGain = -999
        for i in range(x.shape[1]):
            cur = x[:, i]
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
        if node.depth >= self.max_depth:
            node.is_terminal = True
            self.leaves += 1
            return

        if x.shape[0] < self.min_samples_split or x.shape[0] < (int)(1 // self.eta):
            node.is_terminal = True
            self.leaves += 1
            return

        bestSplit, bestThre, x_left, x_right, g_left, g_right, h_left, h_right = self.split(x, g, h)

        if bestSplit is None:
            node.is_terminal = True
            self.leaves += 1
            return

        if x_left.shape[0] < self.min_samples_leaf or x_right.shape[0] < self.min_samples_leaf:
            node.is_terminal = True
            self.leaves += 1
            return

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
        self.Tree.prob = self.NodeProb(g, h)
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
