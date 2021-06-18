from Algorithm.Tree.decision_tree import decision_tree
import numpy as np

from sklearn.datasets import load_iris
import time


def Get_Classifer(name='', **kwargs):
    switcher = {
        'decision_tree': decision_tree
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res(**kwargs)

def to_one_hot(x):
    res = np.zeros((x.size, x.max()+1))
    res[np.arange(x.size), x] = 1
    return res


if __name__ == '__main__':
    data = load_iris()
    x, y, col = data['data'], data['target'], data['feature_names']

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)
    print(X_train.shape)
    model = Get_Classifer('decision_tree')

    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print((end - start) * 100)

    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_val)
    print(y_pred)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')
