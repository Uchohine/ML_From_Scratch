import numpy as np


def Gini(x):
    return 1 - np.sum(x ** 2)


def Entropy(x):
    return - np.sum(x * np.log2(x + np.finfo(float).eps))  # add eps to prevent div by 0 err


def CrossEntropy(ypred, y):
    return - np.sum(y * np.log2(ypred + np.finfo(float).eps))


def Hinge(yHat, y):
    return np.max(0, y - (1 - 2 * y) * yHat)


def Huber(yHat, y, delta=1.):
    return np.where(np.abs(y - yHat) < delta, .5 * (y - yHat) ** 2, delta * (np.abs(y - yHat) - 0.5 * delta))


def KLDivergence(yHat, y):
    """
    :param yHat:
    :param y:
    :return: KLDiv(yHat || y)
    """
    return np.sum(yHat * np.log((yHat / y)))


def L1(yHat, y):
    return np.sum(np.absolute(yHat - y)) / y.size


def MSE(yHat, y):
    return (yHat - y) ** 2


def Set_Criterion(name=''):
    switcher = {
        'gini': Gini,
        'entropy': Entropy
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res

def Set_Loss(name = ''):
    switcher = {
        'CrossEntropy': CrossEntropy,
        'Hinge': Hinge,
        'Huber': Huber,
        'KLDivergence': KLDivergence,
        'L1': L1,
        'MSE': MSE
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res

if __name__ == '__main__':
    name = 'gini';
    criterion = Criterion(name)
    c = np.array([1, 0, 1, 1])
    print(criterion(c, [0, 1]))
