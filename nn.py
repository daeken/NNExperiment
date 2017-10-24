import atexit, json, math, os.path, random

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
			self.weights = json.load(file(self.bstore, 'r'))
		else:
			self.weights = [[random.uniform(-1.0, 1.0) for j in xrange(input.count + 1)] for i in xrange(count)]

		@atexit.register
		def save():
			json.dump(self.weights, file(self.bstore, 'w'))

	def evaluate(self):
		values = self.input.evaluate()
		self.values = [transfer(sum(values[j] * w for j, w in enumerate(self.weights[i][:-1])) + self.weights[i][-1]) for i in xrange(self.count)]
		return self.values

	def train(self, rate, expected=None, errors=None):
		if expected is not None:
			errors = [expected[i] - self.values[i] for i in xrange(self.count)]
		assert errors is not None

		deltas = [errors[i] * transferDerivative(self.values[i]) for i in xrange(self.count)]

		inputs = self.input.values
		ilen = self.input.count

		for i in xrange(self.count):
			delta = deltas[i] * rate
			for j in xrange(ilen):
				self.weights[i][j] += delta * inputs[j]
			self.weights[i][-1] += delta

		if self.input.input is not None:
			prev = self.input
			perrors = [sum(self.weights[j][i] * deltas[j] for j in xrange(self.count)) for i in xrange(prev.count)]
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
