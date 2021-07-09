import numpy as np

def PCA(data, target_dims=None):
    if target_dims == None:
        target_dims = data.shape[1]
    data = data - np.mean(data, axis=0)
    cov = np.cov(data.T) / data.shape[0]
    w,v = np.linalg.eig(cov)
    idx = w.argsort()[::-1][:target_dims]
    return w[idx], v[:, idx]
