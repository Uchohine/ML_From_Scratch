import numpy as np
from Algorithm.PCA.PCA import PCA

class hashtable():
    def __init__(self, data, k, hash = None):
        if hash is None:
            self.hash = np.random.randn(data.shape[1] - 2, k)
        else:
            self.hash = hash
        self.table = dict()
        self.transform = 1 << np.arange(self.hash.shape[1] - 1, -1, -1)
        self._hash(data)

    def _hash(self, key):
        new_key = np.matmul((np.matmul(key[:, 0:-2], self.hash) >= 0).astype('int'), self.transform)
        for i in range(new_key.shape[0]):
            old_val = self.table.get(new_key[i])
            self.table[new_key[i]] = np.vstack((old_val, key[i, -1])) if old_val is not None else key[i, -1]
        for _, values in self.table.items():
            if type(values) == np.ndarray:
                size = values.shape[0]
            else:
                size = 1
            if size > (key.shape[0] // 2):
                self.hash = np.random.randn(self.hash.shape[0], self.hash.shape[1])
                self.table = dict()
                self._hash(key)

    def dispstructure(self):
        for _, values in self.table.items():
            print(_, values.shape)
        print('------------------------------')


    def __getitem__(self, item):
        key = np.matmul(np.matmul(item, self.hash) >= 0, self.transform)
        return self.table.get(key)

class LSH():
    def __init__(self, k, r=8, l=32):
        self.k = k
        self.r = r
        self.l = l
        self.table = list()

    def fit(self, x, y):
        if len(y.shape) != 2:
            y = y.reshape((-1, 1))
        idx = np.arange(x.shape[0]).reshape((-1, 1))
        value, vector = PCA(x, min(self.r, x.shape[1] // 2))
        self.data = np.hstack((x, y, idx))
        for i in range(self.l):
            a = np.random.randn(vector.shape[1], vector.shape[1])
            hash = np.matmul(vector, a)
            self.table.append(hashtable(self.data, self.r, hash=hash))
        return self

    def predict(self, item):
        res = self.table[0][item]
        for i in range(1, self.l):
            tmp = self.table[i][item]
            if tmp is not None:
                res = np.vstack((res, tmp))
        res = res.squeeze()
        res = np.bincount(res.squeeze().astype('int'))
        res = res.argsort()[::-1][:self.k]
        return self.data[res, 0: -1]





if __name__ == '__main__':
    x = np.random.randn(100,100)
    y = (np.random.randn(100,1) > 0).astype('int')
    hash = LSH()
    hash.fit(x,y)