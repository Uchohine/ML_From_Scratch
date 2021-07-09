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
    def __init__(self, k):
        self.k = k
        self.tree = None
        self.list_size = 0

    def buildtree(self, data, node):
        if data.shape[0] <= 3:
            node.is_terminal = True
            node.neighbors = data
            return
        node.column = np.argmax(np.var(data[:, 0:-1], axis=0))
        #this can be np.median, but I find that np.mean gives slightly better performance
        node.threshold = np.mean(data[:, node.column])

        left = data[:, node.column] < node.threshold
        right = data[:, node.column] >= node.threshold

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

    def _prune(self, cand, item, dist, node = Node()):
        if self.visited[node.idx] != 0:
            return cand, dist
        self.visited[node.idx] = 1
        col_dist = abs(cand[-1, node.column] - item[node.column])
        if col_dist > abs(node.threshold - item[node.column]):
            cand, dist = self._predict(cand, item, dist, node.left)
            cand, dist = self._predict(cand, item, dist, node.right)
        if node.parent != None:
            cand, dist = self._prune(cand, item, dist, node.parent)
        return cand, dist

    def _predict(self, cand, item, dist, node):
        if node.is_terminal:
            tmp = list()
            for i in range(node.neighbors.shape[0]):
                tmp.append(np.linalg.norm(item - node.neighbors[i, 0:-1]))
            dist = np.hstack((dist, np.array(tmp)))
            idx = np.argsort(np.array(dist))[0:min(self.k, dist.shape[0])]
            dist = dist[idx]
            self.visited[node.idx] = 1
            cand = np.vstack((cand, node.neighbors))[idx] if cand is not None else node.neighbors[idx]
            pool, dist = self._prune(cand, item, dist, node.parent)
            return pool, dist

        if item[node.column] >= node.threshold:
            return self._predict(cand, item, dist, node.right)

        if item[node.column] < node.threshold:
            return self._predict(cand, item, dist, node.left)

    def predict(self, item):
        cand = None
        dist = np.array(list())
        cand, dist = self._predict(cand, item, dist, self.tree)
        self.visited = (self.list_size + 1) * [0]
        return cand


if __name__ == '__main__':
    x = np.random.randn(100,100)
    y = (np.random.randn(100,1) > 0).astype('int')
    hash = KDTree(k = 3)
    hash.fit(x, y)
