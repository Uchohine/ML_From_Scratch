import numpy as np

def NodeProb(y, classes):
	prob = []
	for c in classes:
		prob.append(y[y == c].shape[0] / y.shape[0])
	return np.array(prob)

def Gini(y, classes):
	return 1 - np.sum(NodeProb(y,classes)**2)

def Entropy(y,classes):
	tmp = NodeProb(y, classes)
	return - np.sum(tmp * np.log2(tmp + np.finfo(float).eps)) #add eps to prevent div by 0 err

def Criterion (name = ''):
	switcher = {
		'gini':Gini,
		'entropy':Entropy
	}
	res = switcher.get(name)
	if res == None:
		raise Exception('No Criterion Named: "{name}".'.format(name = name));
	return res



if __name__ == '__main__':
	name = 'gini';
	criterion = Criterion(name)
	c = np.array([1,0,1,1])
	print(criterion(c,[0,1]))

