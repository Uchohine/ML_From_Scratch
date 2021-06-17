import numpy as np
from Algorithm import util


class Adaboost():
	def __init__(self, terminal_iter = 100, classifier = 'decision_tree'):
		self.classifier = util.Get_Classifer(classifier)
		self.alpha = np.array(list())
		self.terminal = terminal_iter

	def fit(self,x,y):
		self.weights = np.ones(x.shape[0]) / x.shape[0]
		for i in range(self.terminal):
			idx = np.random.choice(self.weights.shape[0], self.weights.shape[0], p = self.weights)

			x_train = x[idx]
			y_train = y[idx]

			self.classifier.fit(x_train, y_train)
			ypred = self.classifier.predict(x_train)

			epi = np.sum((ypred != y_train) * self.weights)


	def __str__(self):
		return self.alpha.__str__()




if __name__ == '__main__':
	c = Adaboost()
	print(c)