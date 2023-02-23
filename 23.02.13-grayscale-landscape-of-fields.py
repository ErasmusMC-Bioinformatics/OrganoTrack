import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# one field
lena = cv.imread("G:/My Drive/mep/data/mask/d0r1t0.tiff", cv.IMREAD_GRAYSCALE)
lena = lena[100:200,100:200]

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

# create the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)

plt.show()