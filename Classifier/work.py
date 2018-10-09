import math

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# mpl.rcParams['legend.fontsize'] = 10
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# ax.plot(x, y, z, label='parametric curve')
# ax.legend()
#
# plt.show()

def count_foil_grow(p0, n0, p, n):
    if p0 == 0 and n0 == 0:
        if p == 0:
            return -math.inf
        try:
            return p * (p / (p + n))
        except (ZeroDivisionError, ValueError):
            return -math.inf
    else:
        if p == 0:
            return -math.inf
        if n == 0 and n0 == 0:
            return p - p0
        try:
            return p * (math.log((p / (p + n)), 2) - math.log((p0 / (p0 + n0)), 2))
        except (ZeroDivisionError, ValueError):
            return -math.inf


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.arange(0, 100, 2)
y = np.arange(0, 100, 2)
X_grid, Y_grid = np.meshgrid(x,y)
Z_grid = X_grid * Y_grid
ax.plot_surface(X_grid, Y_grid, Z_grid, label='parametric curve')
ax.legend()
plt.show()


