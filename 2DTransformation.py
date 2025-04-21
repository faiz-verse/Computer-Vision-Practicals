# 2D Transformation

import numpy as np
import matplotlib.pyplot as plt

# Homogeneous 2D point
point_2d = np.array([[1], [1], [1]])  # (x, y, 1)

def translation(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def rotation(theta):  # Theta in degrees
    theta = np.radians(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

def scaling(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

def apply_transformation(point, transformation):
    return transformation @ point

# Transformation parameters
tx, ty = 2, 3
theta = 45
sx, sy = 2, 3

# Create transformation matrices
T = translation(tx, ty)
R = rotation(theta)
S = scaling(sx, sy)

# Apply transformations
translated_point = apply_transformation(point_2d, T)
rotated_point = apply_transformation(point_2d, R)
scaled_point = apply_transformation(point_2d, S)

# Print results
print("\nOriginal Point (Homogeneous): ", point_2d.flatten())
print("Translated Point: ", translated_point.flatten())
print("Rotated Point: ", rotated_point.flatten())
print("Scaled Point: ", scaled_point.flatten())

# plotting
plt.figure(figsize=(5, 5))
plt.grid(True)
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
# plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

# Original
plt.plot(point_2d[0], point_2d[1], 'ro', label='Original Point (1,1)')

# Transformed points
plt.plot(translated_point[0], translated_point[1], 'bo', label=f'Translated ({tx},{ty})')
plt.plot(rotated_point[0], rotated_point[1], 'go', label=f'Rotated {theta}°')
plt.plot(scaled_point[0], scaled_point[1], 'mo', label=f'Scaled ({sx},{sy})')

# Lines to show movement
plt.plot([point_2d[0][0], translated_point[0][0]], [point_2d[1][0], translated_point[1][0]], 'b--')
plt.plot([point_2d[0][0], rotated_point[0][0]], [point_2d[1][0], rotated_point[1][0]], 'g--')
plt.plot([point_2d[0][0], scaled_point[0][0]], [point_2d[1][0], scaled_point[1][0]], 'm--')

# Labels and plot setup
plt.title('2D Geometric Transformations')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
