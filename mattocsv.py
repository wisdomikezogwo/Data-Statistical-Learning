import scipy.io
import numpy as np

data = scipy.io.loadmat('U001ai.mat')

for i in data:
    if '__' not in i and 'readme' not in i :
        np.savetxt(("eegfile/"+i+".csv"), data[i], delimiter=',')

