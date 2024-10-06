import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# The size of the detector is 1,28 cm x 1,28 cm and has 16x16 pixels
# The thickness of the detector is 0,2 cm
# I want to determine the possible angles of the muons knowing the position of the pixels that have been hit
# I will use the data from the muon detector that has been hit by a muon
# Ths hypothese is that the muons hit the pixels at their centers


thickness = 0.2
size = 1.28
pixels = 16
pixel_size = size/pixels

# create a 16x16 matrix of muons that have hit the detector with 6 pixels hit

muons = np.zeros((pixels,pixels))


muons[0,0] = 1
muons[0,1] = 1
muons[0,2] = 1
muons[0,3] = 1
muons[0,4] = 1
muons[0,5] = 1










