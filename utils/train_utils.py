# This is a Train utils Python script.
# Author: lycaoduong


import numpy as np


a = np.array([0, 0])

x = np.array([[[1 , 2, 3], [2, 1, 5]], [[1 ,2, 6], [4, 3, 5]]])
psum = np.sum(x, axis=(1, 2))
print(psum/6)
print(np.mean(x, axis=(1, 2)))