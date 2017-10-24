from PIL import Image

import json, sys
from nn import *

if sys.argv[1].endswith('json'):
	testset = json.load(file(sys.argv[1], 'rb'))
else:
	im = Image.open(sys.argv[1]).convert('L')
	testset = [[-1, [x / 255. for x in im.getdata()]]]

def toExpected(digit):
	expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	expected[digit] = 1
	return expected

testset = [[digit, pixels, toExpected(digit)] for digit, pixels in testset]

inputs = InputLayer([0] * 784)
hidden1 = Layer(32, inputs, bstore='hidden1')
hidden2 = Layer(16, hidden1, bstore='hidden2')
hidden3 = Layer(16, hidden2, bstore='hidden3')
outputs = Layer(10, hidden3, bstore='outputs')

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

"""
output = evalOne(testset[0][1])
print [int(x * 100) for x in output]
print output.index(sorted(output)[-1])
"""

print 'Training'
for i in xrange(0):
	print 'Training #', i
	for digit, pixels, expected in testset:
		inputs.values = pixels
		outputs.evaluate()
		outputs.train(0.3, expected)

print 'Evaluating'
success = 0
for digit, pixels, _ in testset:
	inputs.values = pixels
	output = evalOne(pixels)
	if digit == output.index(sorted(output)[-1]):
		success += 1

print 'Success rate: %i/%i - %i%%' % (success, len(testset), float(success) / len(testset) * 100)
