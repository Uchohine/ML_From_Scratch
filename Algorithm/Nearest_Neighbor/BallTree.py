import numpy as np

class Node:
    def __init__(self):
        self.parent = None

        # links to the left and right child nodes
        self.right = None
        self.left = None

        # derived from splitting criteria
        self.mean = None
        self.radius = None
        self.projection = None

        # probability for object inside the Node to belong for each of the given classes
        self.neighbors = None
        # depth of the given node
        self.idx = None

        # if it is the root Node or not
        self.is_terminal = False


class BallTree():
    def __init__(self, k, distance_measure):
        self.k = k
        self.tree = None
        self.list_size = 0
        self.distance_measure = distance_measure

    def _find_farthest_pt(self, x, data):
        dist = list()
        for i in range(data.shape[0]):
            dist.append(self.distance_measure(x, data[i, :]))
        dist = np.argsort(np.array(dist))
        return data[dist[-1], :]

    def buildtree(self, data, node = Node()):
        if data.shape[0] <= 3:
            node.neighbors = data
            node.is_terminal = True
            return
        anchor = data[0, 0:-1]
        left = self._find_farthest_pt(anchor, data[:, 0:-1])
        right = self._find_farthest_pt(left, data[:, 0:-1])
        node.projection = right - left
        node.mean = node.projection / 2
        norm = np.linalg.norm(node.projection)
        node.projection = node.projection / norm ** 2
        projection = (np.matmul(node.projection.reshape((1, -1)), data[:, 0:-1].T)).squeeze()
        node.radius = np.mean(projection)
        left = projection < node.radius
        right = projection >= node.radius

        node.left = Node()
        node.left.parent = node
        node.left.idx = 2 * node.idx
        self.buildtree(data[left], node.left)

        node.right = Node()
        node.right.parent = node
        node.right.idx = 2 * node.idx
        self.buildtree(data[right], node.right)

        self.list_size = max(self.list_size, node.right.idx)

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
        col_dist = dist[-1]
        if col_dist > self.distance_measure(node.mean, item):
            cand, dist = self._predict(cand, item, dist, node.left)
            cand, dist = self._predict(cand, item, dist, node.right)
        if node.parent != None:
            cand, dist = self._prune(cand, item, dist, node.parent)
        return cand, dist

    def _predict(self, cand, item, dist, node = Node()):
        if node.is_terminal:
            tmp = list()
            for i in range(node.neighbors.shape[0]):
                tmp.append(self.distance_measure(item, node.neighbors[i, 0:-1]))
            dist = np.hstack((dist, np.array(tmp)))
            idx = np.argsort(np.array(dist))[0:min(self.k, dist.shape[0])]
            dist = dist[idx]
            self.visited[node.idx] = 1
            cand = np.vstack((cand, node.neighbors))[idx] if cand is not None else node.neighbors[idx]
            pool, dist = self._prune(cand, item, dist, node.parent)
            return pool, dist

        projection = (np.matmul(node.projection.reshape((1, -1)), item.T)).squeeze()

        if projection >= node.radius:
            return self._predict(cand, item, dist, node.right)

        if projection < node.radius:
            return self._predict(cand, item, dist, node.left)

    def predict(self, item):
        cand = None
        dist = np.array(list())
        cand, dist = self._predict(cand, item, dist, self.tree)
        self.visited = (self.list_size + 1) * [0]
        return cand

