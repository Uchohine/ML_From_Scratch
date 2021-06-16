import numpy as np

import Gini_Impurity

def Criterion (name = ''):
	switcher = {
		'gini':Gini_Impurity.Gini_Impurity
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

