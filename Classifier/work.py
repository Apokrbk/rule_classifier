from pyroaring import BitMap

x = [0,2,5,6,9,11]
y = BitMap(x)
print(y[2])
