# 3D Transformation

import numpy as np
import matplotlib.pyplot as plt

point_3d = np.array([[1], [1], [1], [1]])  # Homogeneous coordinate

# Translation Matrix
def translation(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

# Rotation Matrix (X-axis)
def rotation_x(theta):
    theta = np.radians(theta)  # Convert to radians
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

# Rotation Matrix (Y-axis)
def rotation_y(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

# Rotation Matrix (Z-axis)
def rotation_z(theta):
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Scaling Matrix
def scaling(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

# Function to apply transformation
def apply_transformation(point, transformation):
    transformed_point = transformation @ point  # Corrected multiplication
    return transformed_point

# Define transformation parameters
tx, ty, tz = 1, 2, 3      # Translation
theta_x, theta_y, theta_z = 45, 30, 60  # Rotation angles in degrees
sx, sy, sz = 2, 2, 2      # Scaling factors

# Compute transformation matrices
T = translation(tx, ty, tz)
R_x = rotation_x(theta_x)
R_y = rotation_y(theta_y)
R_z = rotation_z(theta_z)
S = scaling(sx, sy, sz)

# Apply transformations
translated_point = apply_transformation(point_3d, T)
rotated_x_point = apply_transformation(point_3d, R_x)
rotated_y_point = apply_transformation(point_3d, R_y)
rotated_z_point = apply_transformation(point_3d, R_z)
scaled_point = apply_transformation(point_3d, S)

# Print results
print("\nOriginal Point (Homogeneous): ", point_3d.flatten())
print("Translated Point: ", translated_point.flatten())
print("Rotated around X-axis: ", rotated_x_point.flatten())
print("Rotated around Y-axis: ", rotated_y_point.flatten())
print("Rotated around Z-axis: ", rotated_z_point.flatten())
print("Scaled Point: ", scaled_point.flatten())

# PLOTTING
# Extract XYZ coordinates from transformed points (helper function)
def extract_coords(point):
    return point[0, 0], point[1, 0], point[2, 0]

# Get all transformed points
x0, y0, z0 = extract_coords(point_3d)
xt, yt, zt = extract_coords(translated_point)
xs, ys, zs = extract_coords(scaled_point)
xrx, yrx, zrx = extract_coords(rotated_x_point)
xry, yry, zry = extract_coords(rotated_y_point)
xrz, yrz, zrz = extract_coords(rotated_z_point)

# Plotting
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
