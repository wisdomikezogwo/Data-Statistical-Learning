from sklearn import datasets
from sklearn.decomposition import PCA as sklearn_PCA
import matplotlib.pyplot as plt
iris = datasets.load_iris()
#from mpl_toolkits.mplot3d import Axes3D
import  numpy as np

X = iris.data
Y = iris.target
target_names = iris.target_names



np.random.seed(5)
skPCA = sklearn_PCA(n_components=2)
X_learn = skPCA.fit(X).transform(X)

plt.figure()
colours =['navy', 'turquoise', 'red']
lw = 2

for color, i , target_name in zip(colours, [0,1,2], target_names):
    plt.scatter(X_learn[Y ==i, 0], X_learn[Y ==i,1],

                color = color, alpha =.8, lw=lw, label = target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()