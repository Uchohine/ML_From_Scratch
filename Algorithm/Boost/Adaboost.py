import numpy as np
from Algorithm import util

from sklearn.datasets import load_wine
import time


class Adaboost():
	def __init__(self, terminal_iter = 50, classifier = 'decision_tree', **kwargs):
		self.classifier = classifier
		self.classifiers = list()
		self.alpha = list()
		self.terminal = terminal_iter
		self.arg = kwargs

	def fit(self, x, y):
		self.weights = np.ones(x.shape[0]) / x.shape[0]
		for i in range(self.terminal):
			idx = np.random.choice(self.weights.shape[0], self.weights.shape[0], p=self.weights)
			x_train = x[idx]
			y_train = y[idx]

			model = util.Get_Classifer(self.classifier, **self.arg)
			model.no_grad = True
			model.fit(x_train, y_train)
			ypred = model.predict(x_train)
			err = (ypred != y_train)

			epi = np.matmul(self.weights.T, err)
			if epi == .5 or epi == 0:
				break

			alpha = .5 * np.log((1/epi) - 1)
			self.alpha.append(alpha)
			self.classifiers.append(model)

			self.weights *= np.exp(-alpha * (err + np.finfo(float).eps))
			self.weights /= np.sum(self.weights)
		self.alpha = np.array(self.alpha)
		self.alpha = self.alpha / np.sum(self.alpha)

	def predict(self, x):
		y = list()
		for i in range(len(self.alpha)):
			y.append(self.classifiers[i].predict(x))
		y = np.array(y)
		return np.rint(np.matmul(y.T, self.alpha))

	def __str__(self):
		return np.array2string(self.alpha)




if __name__ == '__main__':
	data = load_wine()
	x, y, col = data['data'], data['target'], data['feature_names']

	from sklearn.model_selection import train_test_split

	X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=44)

	model = Adaboost(max_depth=10)

	start = time.time()
	model.fit(X_train, y_train)
	end = time.time()
	print('elapsed time : {:.2f}ms'.format((end - start)*1000))

	from sklearn.metrics import accuracy_score
	y_pred = model.predict(X_val)
	print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

	from sklearn.ensemble import AdaBoostClassifier

	model = AdaBoostClassifier(n_estimators=50, random_state=0)
	start = time.time()
	model.fit(X_train, y_train)
	end = time.time()
	print('elapsed time : {:.2f}ms'.format((end - start)*1000))
	y_pred = model.predict(X_val)
	print(f'Accuracy for sklearn Decision Tree {accuracy_score(y_val, y_pred)}')
