import json, struct, sys

lfp = file(sys.argv[1], 'rb')
ifp = file(sys.argv[2], 'rb')

assert struct.unpack('>I', lfp.read(4))[0] == 2049
assert struct.unpack('>I', ifp.read(4))[0] == 2051

count, = struct.unpack('>I', lfp.read(4))
assert struct.unpack('>I', ifp.read(4))[0] == count
assert struct.unpack('>I', ifp.read(4))[0] == 28
assert struct.unpack('>I', ifp.read(4))[0] == 28

data = []
for i in xrange(count):
	digit = ord(lfp.read(1))
	pixels = [ord(x) / 255. for x in ifp.read(28*28)]
	data.append((digit, pixels))

json.dump(data, file(sys.argv[3], 'w'))
