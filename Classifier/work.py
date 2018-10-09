from pyroaring import BitMap

x = [0,2,5,6]
y = BitMap(x)
print(y)

print(list(y)[2])