import numpy as np
import numpy.matlib


from sklearn.datasets import load_wine
import time

class kmean():
    def __init__(self, k=10, initial='kmean++', T=50000):
        self.k = k
        self.initial = initial
        self.T = T

    def _init(self, x, size, p=None):
        return x[np.random.choice(np.arange(x.shape[0]), size=size, p=p), :]

    def _kmeanplus_init(self, x, size):
        mean = np.zeros((size, x.shape[1]))
        mean[0, :] = x[0, :]
        weight = np.sum((x - np.matlib.repmat(mean[0, :], x.shape[0], 1)) ** 2, axis=1)
        for i in range(1, size):
            mean[i, :] = self._init(x, 1, p=weight/np.sum(weight))
            weight = np.min(np.vstack((weight, np.sum((x - np.matlib.repmat(mean[i, :], x.shape[0], 1)) ** 2, axis=1))),
                            axis=0)
        return mean

    def fit(self, x):
        if self.initial == 'random':
            self.mean = self._init(x, self.k)
        if self.initial == 'kmean++':
            self.mean = self._kmeanplus_init(x, self.k)
        last = np.zeros(x.shape[0])
        for i in range(self.T):
            print(i)
            distance_matrix = np.zeros((self.k, x.shape[0]))
            for j in range(self.k):
                distance_matrix[j, :] = np.sum((x - np.matlib.repmat(self.mean[j, :], x.shape[0], 1)) ** 2, axis=1)
            idx = np.argmin(distance_matrix, axis=0)
            if (last == idx).all():
                break
            else:
                last = idx
                for j in range(self.k):
                    tmp = (idx == j).astype('float64').reshape(1, -1)
                    self.mean[j, :] = np.matmul(tmp, x) / np.sum(tmp, axis=1)
        return self.mean





if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)
    model = kmean(k=3)

    start = time.time()
    model.fit(X_train)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))

