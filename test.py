from PIL import Image

import json, sys
from nn import *

if sys.argv[1].endswith('json'):
	testset = json.load(file(sys.argv[1], 'rb'))
else:
	im = Image.open(sys.argv[1]).convert('L')
	testset = [[-1, [x / 255. for x in im.getdata()]]]

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

output = evalOne(testset[0][1])
print [int(x * 100) for x in output]
print output.index(sorted(output)[-1])

"""print 'Training'
for i in xrange(1):
	print 'Training #', i
	for digit, pixels in testset:
		inputs.values = pixels
		expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		expected[digit] = 1
		outputs.evaluate()
		outputs.train(0.3, expected)

print 'Evaluating'
success = 0
for digit, pixels in testset:
	inputs.values = pixels
	output = evalOne(pixels)
	if digit == output.index(sorted(output)[-1]):
		success += 1

print 'Success rate: %i/%i - %i%%' % (success, len(testset), float(success) / len(testset) * 100)"""
