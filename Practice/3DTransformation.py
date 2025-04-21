import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import projection_registry
from matplotlib.pyplot import figure

point_3d = np.array([[1], [1], [1], [1]])


# translation
def translation(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


# scaling
def scaling(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


# rotation (rotate x)
def rotate_x(theta):
    theta = np.radians(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])


# rotation (rotate y)
def rotate_y(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])


# rotation (rotate z)
def rotate_z(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def applyTransformation(point, transformation):
    return transformation @ point


tx, ty, tz = 1, 2, 3
sx, sy, sz = 1, 2, 3
theta_x, theta_y, theta_z = 45, 30, 60

T = translation(tx, ty, tz)
S = scaling(sx, sy, sz)
Rx = rotate_x(theta_x)
Ry = rotate_y(theta_y)
Rz = rotate_z(theta_z)

translatedPoints = applyTransformation(point_3d, T)
scaledPoints = applyTransformation(point_3d, S)
rotated_x = applyTransformation(point_3d, Rx)
rotated_y = applyTransformation(point_3d, Ry)
rotated_z = applyTransformation(point_3d, Rz)

# Print results
print("\nOriginal Point (Homogeneous): ", point_3d.flatten())
print("Translated Point: ", translatedPoints.flatten())
print("Scaled Point: ", scaledPoints.flatten())
print("Rotated around X-axis: ", rotated_x.flatten())
print("Rotated around Y-axis: ", rotated_y.flatten())
print("Rotated around Z-axis: ", rotated_z.flatten())


# Plotting
# Helper function to extract the coords from the points
def extractCoords(point):
    return point[0, 0], point[1, 0], point[2, 0]


# getting coords
x0, y0, z0 = extractCoords(point_3d)
xt, yt, zt = extractCoords(translatedPoints)
xs, ys, zs = extractCoords(scaledPoints)
xrx, yrx, zrx = extractCoords(rotated_x)
xry, yry, zry = extractCoords(rotated_y)
xrz, yrz, zrz = extractCoords(rotated_z)

# plotting
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each point with a label
ax.scatter(x0, y0, z0, color='red', label='Original (1,1,1)')
ax.scatter(xt, yt, zt, color='blue', label='Translated')
ax.scatter(xs, ys, zs, color='magenta', label='Scaled')
ax.scatter(xrx, yrx, zrx, color='green', label='Rotated X-axis')
ax.scatter(xry, yry, zry, color='orange', label='Rotated Y-axis')
ax.scatter(xrz, yrz, zrz, color='purple', label='Rotated Z-axis')

# Connect original to others with lines
ax.plot([x0, xt], [y0, yt], [z0, zt], color='blue', linestyle='dashed')
ax.plot([x0, xs], [y0, ys], [z0, zs], color='magenta', linestyle='dashed')
ax.plot([x0, xrx], [y0, yrx], [z0, zrx], color='green', linestyle='dashed')
ax.plot([x0, xry], [y0, yry], [z0, zry], color='orange', linestyle='dashed')
ax.plot([x0, xrz], [y0, yrz], [z0, zrz], color='purple', linestyle='dashed')

# Set labels and legend
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Geometric Transformations of a Point')
ax.legend()

# Set the same scale for better visualization
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

plt.show()
