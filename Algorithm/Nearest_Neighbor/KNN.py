import numpy as np
from Algorithm.Nearest_Neighbor.LSH import LSH
from Algorithm.Nearest_Neighbor.KDTree import KDTree
from Algorithm.Nearest_Neighbor.BallTree import BallTree
from Distance_Measure.Distance import get_distance_measure

import pandas as pd
from sklearn.datasets import load_wine
import time

class KNN():
    def __init__(self, algorithm='LSH', k=1, distance_measure='cos', **kwargs):
        self.algorithm = algorithm
        self.k = k
        self.arg = kwargs
        self.distance_measure = get_distance_measure(distance_measure)
        self.db = None

    def fit(self,x,y):
        if self.algorithm == 'LSH':
            self.db = LSH(k=self.k).fit(x,y)
        if self.algorithm == 'KDTree':
            self.distance_measure = get_distance_measure('l2')
            self.db = KDTree(self.k).fit(x,y)
        if self.algorithm == 'BallTree':
            #although BallTree can use cos distance theoritically, but the projection calculation is based on l2 norm.
            #I will fix distance measurement on l2 until I find some reference if I can do cos.
            self.distance_measure = get_distance_measure('l2')
            self.db = BallTree(self.k, self.distance_measure).fit(x,y)
        if self.algorithm == 'BruteForce':
            if len(y.shape) != 2:
                y = y.reshape(-1,1)
            self.db = np.hstack((x, y))
        return self

    def predict(self, item):
        pred = list()
        for i in range(item.shape[0]):
            cur = item[i, :]
            if self.algorithm == 'BruteForce':
                nn = self.db
            else:
                nn = self.db.predict(cur)
            if nn.shape[0] > self.k:
                print(nn.shape)
                dist = list()
                for i in range(nn.shape[0]):
                    dist.append(self.distance_measure(nn[i, 0:-1], cur))
                idx = np.argsort(np.array(dist))
                idx = idx[0:min(self.k, idx.shape[0])]
                pred.append(np.argmax(np.bincount(nn[idx, -1].astype('int'))))
            else:
                pred.append(np.argmax(np.bincount(nn[:, -1].astype('int'))))
        return np.array(pred)

if __name__ == '__main__':
    data = load_wine()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)
    model = KNN(k=15, distance_measure='cos', algorithm='LSH')

    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    start = time.time()
    y_pred = model.predict(X_val)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))
    print(y_pred)
    print(y_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=15, algorithm='ball_tree')


    model.fit(X_train, y_train)
    start = time.time()
    y_pred_ref = model.predict(X_val)
    end = time.time()
    print('elapsed time : {:.5f}s'.format((end - start)))
    print(y_pred_ref)
    print(f'Accuracy for sklearn KNN {accuracy_score(y_val, y_pred_ref)}')




