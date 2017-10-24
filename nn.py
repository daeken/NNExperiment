import atexit, json, math, os.path, random
import numpy as np

def sigmoid(x):
	return 1.0 / (math.exp(-x) + 1)

def sigmoidDerivative(x):
	return x * (1.0 - x)

transfer = sigmoid
transferDerivative = sigmoidDerivative

class Layer(object):
	def __init__(self, count, input, bstore):
		self.count = count
		self.input = input
		self.values = [0] * count
		self.bstore = bstore + '.json'
		if os.path.exists(self.bstore):
			w, b = json.load(file(self.bstore, 'r'))
			self.weights, self.biases = np.matrix(w), np.array(b)
		else:
			self.weights = np.matrix([[random.uniform(-1.0, 1.0) for j in xrange(input.count)] for i in xrange(count)])
			self.biases = np.array([[random.uniform(-1.0, 1.0)] for j in xrange(count)])

		@atexit.register
		def save():
			json.dump((self.weights.tolist(), self.biases.tolist()), file(self.bstore, 'w'))

	def evaluate(self):
		values = self.input.evaluate()
		self.values = map(transfer, self.weights * np.array(values).reshape((self.input.count, 1)) + self.biases)
		return self.values

	def train(self, rate, expected=None, errors=None):
		if expected is not None:
			errors = np.array([expected[i] - self.values[i] for i in xrange(self.count)]).reshape((self.count, 1))
		assert errors is not None

		deltas = np.multiply(np.array(map(transferDerivative, self.values)).reshape((self.count, 1)), errors) * rate

		inputs = self.input.values
		ilen = self.input.count

		self.weights += deltas * inputs
		self.biases += deltas

		if self.input.input is not None:
			prev = self.input
			perrors = self.weights.transpose() * deltas
			self.input.train(rate, errors=perrors)

class InputLayer(object):
	def __init__(self, values):
		self.values = values
		self.count = len(values)
		self.input = None

	def evaluate(self):
		return self.values

def cost(a, b):
	return sum((a[i] - b[i]) ** 2 for i in xrange(len(a)))
