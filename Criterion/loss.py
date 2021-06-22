import numpy as np


def softmax(x):
    tmp = np.exp(x.T - np.amax(x, axis=1))
    tmp = (tmp / np.sum(tmp, axis=0)).T
    return tmp

class CrossEntropy():
    def __init__(self, **kwargs):
        self.name = 'CrossEntropy'

    def forward(self, y, ypred):
        return - np.sum(y * np.log(ypred + np.finfo(float).eps))

    def Gradient(self, y, ypred):
        return -(y - softmax(ypred))

    def Hessian(self, y, ypred):
        p = softmax(ypred)
        return p * (1 - p)

class MSE():
    def __init__(self):
        self.name = 'MSE'

    def forward(self, y, ypred):
        return (y - ypred) ** 2

    def Gradient(self, y, ypred):
        return - (y - ypred)

    def Hessian(self, y, ypred):
        return np.ones(ypred.shape)

class Deviance():
    def __init__(self):
        self.name = 'devicance'

    def forward(self, y, ypred):
        return - np.sum(y * np.log(softmax(ypred)))

    def Gradient(self, y, ypred):
        return -(y - softmax(ypred))


def Set_Loss(name='', **kwargs):
    switcher = {
        'CrossEntropy': CrossEntropy,
        'Deviance': Deviance,
        'MSE':MSE
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res(**kwargs)
