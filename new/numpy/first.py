import numpy as np
import time
import sys

l = range(1000000)
l2 = range(1000000)

print(sys.getsizeof(5)*len(l))



arr = np.arange(1000000)
arr1 = np.arange(1000000)
print(arr.size*arr.itemsize)

#to measure the time

start = time.time()
result = [(x+y) for x,y in zip(l,l2)]

print("time : ", (time.time()-start)*1000 )

start = time.time()
result1 = arr + arr1
print("time : ", (time.time()-start)*1000)