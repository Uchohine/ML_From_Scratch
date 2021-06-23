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

    def Weighted_Quantile_Sketch(self, pos, weight):
        idx = np.argsort(pos)
        pos = pos[idx]
        weight = weight[idx]
        skip = np.sum(weight) * self.eta
        candidate = list()
        err = 0
        for i in range(weight.shape[0]):
            err += np.sum(weight[i, :])
            if err >= skip:
                candidate.append(pos[i])
                err = 0
        return candidate

    def Sparse_Separate(self, x, g, h):
        is_nan = np.isnan(x)
        not_nan = ~is_nan
        valid_cur = x[not_nan]
        valid_g = g[not_nan]
        valid_h = h[not_nan]
        nan_g = g[is_nan]
        nan_h = h[is_nan]
        return valid_cur, valid_g, valid_h, nan_g, nan_h

    def split(self, x, g, h):
        last = self.Score(g, h)
        bestSplit = None
        bestThre = None
        bestGain = 0
        direction = None
        for i in range(x.shape[1]):
            cur = x[:, i]
            valid_cur, valid_g, valid_h, nan_g, nan_h = self.Sparse_Separate(cur, g, h)
            if self.eta < 1:
                can = self.Weighted_Quantile_Sketch(valid_cur, valid_h)
            else:
                can = valid_cur
            for val in can:
                left = (valid_cur <= val)
                right = (valid_cur > val)
                if nan_g.shape[0] != 0:
                    score = .5 * (self.Score(np.concatenate((valid_g[left], nan_g)),
                                             np.concatenate((valid_h[left], nan_h))) + self.Score(g[right],
                                                                                                  h[
                                                                                                      right]) - last) - self.gamma
                    if score > bestGain:
                        bestSplit = i
                        bestThre = val
                        bestGain = score
                        direction = 0
                score = .5 * (self.Score(valid_g[left], valid_h[left]) + self.Score(np.concatenate((valid_g[right], nan_g)),
                                                                                    np.concatenate((valid_h[right],
                                                                                               nan_h))) - last) - self.gamma
                if score > bestGain:
                    bestSplit = i
                    bestThre = val
                    bestGain = score
                    direction = 1

        if bestGain == 0:
            return None, None, None, None, None, None, None, None, None

        idx = x[:, bestSplit]
        x_left, x_right = x[idx <= bestThre, :], x[idx > bestThre, :]
        g_left, g_right = g[idx <= bestThre, :], g[idx > bestThre, :]
        h_left, h_right = h[idx <= bestThre, :], h[idx > bestThre, :]

        return bestSplit, bestThre, direction, x_left, x_right, g_left, g_right, h_left, h_right

    def buildtree(self, x, g, h, node):
        if node.depth >= self.max_depth:
            node.is_terminal = True
            self.leaves += 1
            return

        if x.shape[0] < self.min_samples_split or x.shape[0] < (int)(1 // self.eta):
            node.is_terminal = True
            self.leaves += 1
            return

        bestSplit, bestThre, direction, x_left, x_right, g_left, g_right, h_left, h_right = self.split(x, g, h)

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
        node.direct = direction

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

        if np.isnan(x[node.column]):
            if node.direct:
                prob = self.predictSample(x, node.right)
            else:
                prob = self.predictSample(x, node.left)
        elif x[node.column] > node.threshold:
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
