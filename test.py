import json, sys
from nn import *

testset = json.load(file(sys.argv[1], 'rb'))

inputs = InputLayer([0] * 768)
hidden1 = Layer(16, inputs, bstore='hidden1')
hidden2 = Layer(16, hidden1, bstore='hidden2')
outputs = Layer(10, hidden2, bstore='outputs')

def evalOne(data):
	inputs.values = data
	return outputs.evaluate()

def costAll():
	total = 0
	for digit, pixels in testset:
		expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expected[digit] = 1
		total += cost(evalOne(pixels), expected)
	return total / len(testset)

"""print 'Training'
for i in xrange(500):
	print 'Training #', i
	for digit, pixels in testset:
		inputs.values = pixels
		expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expected[digit] = 1
		outputs.evaluate()
		outputs.train(expected)
		outputs.update(0.3)"""

print 'Evaluating'
success = 0
for digit, pixels in testset:
	inputs.values = pixels
	output = evalOne(pixels)
	if digit == output.index(sorted(output)[-1]):
		success += 1

print 'Success rate: %i/%i - %i%%' % (success, len(testset), float(success) / len(testset) * 100)
