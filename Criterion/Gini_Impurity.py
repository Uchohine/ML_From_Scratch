#A numpy only implementation of criterion gini index
import numpy as np

def NodeProb(y, classes):
	prob = []
	for c in classes:
		prob.append(y[y == c].shape[0] / y.shape[0])
	return np.array(prob)

def Gini_Impurity(y, classes):
	return 1 - np.sum(NodeProb(y,classes)**2)
	

if __name__ == '__main__':
	a = np.array([[0, 1, 0.2], [0, 0, 0.3],[1, 1, 0.15], [1, 0, 0.2]])
	c = np.array([1,0,1,1])
	print(Gini_Impurity(c, list(set(c.tolist()))))