import matplotlib.pyplot  as plt



x=[[1, 2],[2, 4],[3, 3],[3, 6],[4, 5]]
f=[]
s=[]
for arr in x:
    a, b = arr
    f.append(a)
    s.append(b)
    print a,b
print f,s
plt.plot(f, s, 'ro')

plt.show()
