import numpy as np

class Node:
    def __init__(self):
        self.parent = None

        # links to the left and right child nodes
        self.right = None
        self.left = None

        # derived from splitting criteria
        self.column = None
        self.threshold = None

        # probability for object inside the Node to belong for each of the given classes
        self.neighbors = None
        # depth of the given node
        self.idx = None

        # if it is the root Node or not
        self.is_terminal = False

class KDTree():
    def __init__(self, k, distance_measure):
        self.k = k
        self.tree = None
        self.list_size = 0
        self.distance_measure = distance_measure

    def buildtree(self, data, node):
        if data.shape[0] <= self.k:
            node.is_terminal = True
            node.neighbors = data
            return
        node.column = np.argmax(np.var(data[:, 0:-1], axis=0))
        node.threshold = np.median(data[:, node.column])

        left = data[:, node.column] > node.threshold
        right = data[:, node.column] <= node.threshold

        node.left = Node()
        node.left.parent = node
        node.left.idx = 2 * node.idx

        node.right = Node()
        node.right.parent = node
        node.right.idx = 2 * node.idx + 1

        self.list_size = max(self.list_size, node.right.idx)

        self.buildtree(data[left], node.left)
        self.buildtree(data[right], node.right)

    def fit(self, x, y):
        if len(y.shape) != 2:
            y = y.reshape((-1,1))
        data = np.hstack((x,y))
        self.tree = Node()
        self.tree.idx = 1
        self.buildtree(data, self.tree)
        self.visited = (self.list_size + 1) * [0]
        return self

    def _prune(self, cand, item, node = Node()):
        if self.visited[node.idx] != 0:
            return cand
        self.visited[node.idx] = 1
        if node.is_terminal:
            cand = np.vstack((cand, node.neighbors))
            dist = list()
            for i in range(cand.shape[0]):
                dist.append(self.distance_measure(item, cand[i, 0:-1]))
            idx = np.argsort(np.array(dist))[0:min(self.k, len(dist))]
            return cand[idx]
        else:
            dist = self.distance_measure(cand[-1, node.column], item[node.column])
            if dist > self.distance_measure(node.threshold, item[node.column]):
                cand = self._prune(cand, item, node.left)
                cand = self._prune(cand, item, node.right)
            if node.parent != None:
                cand = self._prune(cand, item, node.parent)
            return cand




    def _predict(self, item, node = Node()):
        if node.is_terminal:
            dist = list()
            for i in range(node.neighbors.shape[0]):
                dist.append(self.distance_measure(item, node.neighbors[i, 0:-1]))
            idx = np.argsort(np.array(dist))[0:min(self.k, len(dist))]
            self.visited[node.idx] = 1
            pool = self._prune(node.neighbors[idx], item, node.parent)
            self.visited = (self.list_size + 1) * [0]
            return pool

        if item[node.column] > node.threshold:
            return self._predict(item, node.right)

        if item[node.column] < node.threshold:
            return self._predict(item, node.left)

    def predict(self, item):
        return self._predict(item, self.tree)


if __name__ == '__main__':
    x = np.random.randn(100,100)
    y = (np.random.randn(100,1) > 0).astype('int')
    hash = KDTree(k = 3)
    hash.fit(x, y)
