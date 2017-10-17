import scipy.io
import pandas as pd
mat = scipy.io.loadmat('U001ai.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.iteritems()})
data = data.to_dense()
data.to_csv("example.csv")