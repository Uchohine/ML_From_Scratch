import numpy as np


class CrossEntropy():
    def __init__(self, **kwargs):
        self.name = 'CrossEntropy'

    def forward(self, y, ypred):
        return - y * np.log2(ypred + np.finfo(float).eps)

    def backward(self, y, ypred):
        tmp = np.exp(ypred.T - np.amax(ypred, axis = 1))
        tmp = (tmp / np.sum(tmp, axis=0)).T
        return y - tmp


def Set_Loss(name='', **kwargs):
    switcher = {
        'CrossEntropy': CrossEntropy,
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res(**kwargs)
