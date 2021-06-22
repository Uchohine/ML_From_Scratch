from Algorithm.Tree.decision_tree import decision_tree
import numpy as np

def Get_Classifer(name='', **kwargs):
    switcher = {
        'decision_tree': decision_tree
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res(**kwargs)

def to_one_hot(x):
    if len(x.shape) != 1:
        return x, x.shape[1]
    res = np.zeros((x.size, x.max()+1))
    res[np.arange(x.size), x] = 1
    return res, res.shape[1]
