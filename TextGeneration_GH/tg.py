import sys


for i in range(10000000):
    sys.stdout.write("\rDoing thing %i" % i)
    sys.stdout.flush()