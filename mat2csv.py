# References:
# https://scipy.github.io/old-wiki/pages/Cookbook/Reading_mat_files.html
# https://github.com/PyHOGS/pyhogs-code/blob/master/notebooks/examples/Reading_complex_mat_files.ipynb

import os
import numpy as np, h5py
import pandas as pd
from scipy.io import loadmat

matdata = loadmat('U001ai.mat')
type(matdata)
print matdata.keys()

# Beginning at release 7.3 of Matlab, mat files are actually saved using the HDF5 format by default
# Not in our case
# import tables
# x = tables.openFile(dataset_train)
# type (x)

# Adapted from https://github.com/PyHOGS/pyhogs-code/tree/master/notebooks
def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """
    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys>0:
        d = {k: d[k] for k in d.keys()[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key, value in d.iteritems():         # iteritems loops through key, value pairs
            print '\t' * indent + 'Key: ' + str(key)
            print_mat_nested(value, indent+1)

    if isinstance(d,np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
        for n in d.dtype.names:    # This means it's a struct, it's bit of a kludge test.
            print '\t' * indent + 'Field: ' + str(n)
            print_mat_nested(d[n], indent+1)

print_mat_nested(matdata, nkeys=10000)