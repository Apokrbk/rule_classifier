from pyroaring import BitMap
import numpy as np

arr = list()
arr.append(False)
arr.append(False)
arr.append(True)
a = np.array(arr)
arr2 = list()
arr2.append(True)
arr2.append(False)
arr2.append(True)
a2 = np.array(arr2)
print(np.logical_or(a,a2))
