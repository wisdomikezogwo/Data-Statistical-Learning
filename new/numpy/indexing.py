import numpy as np

a = np.array([1,2,3,4,5,6])

print a[0:2]
b = np.array([[2,3,4],[5,6,7],[8,9,10]])
print b

print b[1:3,1]
print b[-1,0:2]

print b[:,1:3]

for row in b:
    print row

print b.ravel()

for cell in b.flat:
    print cell
r = np.arange(9).reshape(3,3)
r1 = np.arange(9,18).reshape(3,3)
print r1
print r

# to stack them
print np.vstack((r,r1))
print np.hstack((r,r1))

print np.hsplit(r1,3)


a = np.arange(12).reshape(3,4)
print a

b = a > 4
print b



a[b] = -1
print a




