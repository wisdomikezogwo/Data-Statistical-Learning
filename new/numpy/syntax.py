import numpy as np

a = np.array([[2,3],[4,5]])
print a.ndim
print a.itemsize
print a.dtype

b = np.array([[2,3],[4,5]],dtype=np.float32)

print b.dtype
print b.itemsize

print   b.size,b.shape

c = np.array([[2,3],[4,5]],dtype=complex)

print c

d = np.zeros((3,3))

f= np.ones((3,3))

print d,f

print np.arange(0,5,2)

print np.linspace(1,10,100)

print a.reshape(4,1)

print f.ravel( )
print a

#mathematical functions

print a.min()
print a.sum(axis=0)
print a.sum(axis=1)

print "   "

print np.sqrt(b)
print np.std(b)

e=np.array([[1,2],[3,4]])
g=np.array([[5,6],[7,8]])

print e+g
print e*g
print e
print g
print "   "
print e.dot(g)


