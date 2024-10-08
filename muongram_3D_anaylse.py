import numpy as np
import matplotlib.pyplot as plt

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

matrix = np.zeros((pixels,pixels))

# Calculate the positions of the centers of every pixel
x_centers = np.linspace(pixel_size / 2, size - pixel_size / 2, pixels)
y_centers = np.linspace(pixel_size / 2, size - pixel_size / 2, pixels)
centers = np.array(np.meshgrid(x_centers, y_centers)).T.reshape(-1, 2)

# Plot the centers of the pixels
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color = 'black')
plt.title('Centers of the Pixels')
plt.xlabel('X Position [cm]')
plt.ylabel('Y Position [cm]')
plt.grid(True)
plt.show()

# Create a frame exemple with 1 and 0 
# change y axis of matrix to 1
matrix[5,5:15] = 1 


# Show the matrix and invert y axis
plt.imshow(matrix.T, cmap='gray')
plt.gca().invert_yaxis()
plt.title('Muon Detector')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.show()

z_max = np.max(np.where(matrix !=0 )[1])
z_min = np.min(np.where(matrix !=0 )[1])
z_vector = (z_max - z_min) * pixel_size

x_max = np.max(np.where(matrix !=0 )[0])
x_min = np.min(np.where(matrix !=0 )[0])
x_vector = (x_max - x_min) * pixel_size

y_vector = thickness

rho = np.sqrt(x_vector**2 + y_vector**2 + z_vector**2)
theta = np.arccos(z_vector/rho)
phi = np.arctan(np.sqrt(x_vector**2 + y_vector**2)/z_vector)

print('The norm is', rho)
print('The angle theta is:', np.rad2deg(theta))
print('The angle phi is:', np.rad2deg(phi))

