import numpy as np

def l2(x, y):
    return np.linalg.norm(x - y)

def cosine_distance(a, b):
    print(a)
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

def get_distance_measure(name):
    switcher = {
        'l2': l2,
        'cos': cosine_distance
    }
    res = switcher.get(name)
    if res == None:
        raise Exception('No Criterion Named: "{name}".'.format(name=name));
    return res