import numpy as np


class hashtable():
    def __init__(self, data, label, k, hash = None):
        if hash == None:
            self.hash = np.random.randn(data.shape[1], k)
        else:
            self.hash = hash
        self.table = dict()
        self.transform = 1 << np.arange(k - 1, -1, -1)
        self._hash(data, label)

    def _hash(self, key, value):
        new_key = np.matmul(np.matmul(key, self.hash) >= 0, self.transform)
        if len(value.shape) != 2:
            value = value.reshape((-1, 1))
        key = np.hstack((key, value))
        for i in range(new_key.shape[0]):
            old_val = self.table.get(new_key[i])
            self.table[new_key[i]] = np.vstack((old_val, key[i, :])) if old_val is not None else key[i, :]

    def __getitem__(self, item):
        key = np.matmul(np.matmul(item, self.hash) >= 0, self.transform)
        return self.table.get(key)

class LSH():
    def __init__(self, k=8, l=8):
        self.k = k
        self.l = l
        self.table = list()

    def fit(self, x, y):
        for i in range(self.l):
            self.table.append(hashtable(x,y,self.k))
        return self

    def predict(self, item):
        res = self.table[0][item]
        for i in range(1, self.l):
            res = np.vstack((self.table[i][item], res))
        return res





if __name__ == '__main__':
    x = np.random.randn(100,100)
    y = (np.random.randn(100,1) > 0).astype('int')
    hash = LSH()
    hash.fit(x,y)