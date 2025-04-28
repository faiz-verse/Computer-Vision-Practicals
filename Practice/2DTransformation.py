import numpy as np
import matplotlib.pyplot as plt

# homogenous points
point_2d = np.array([[1], [1], [1]])


# translation
def translate(tx, ty):
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ]
    )


# scale
def scale(sx, sy):
    return np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ]
    )


# rotate
def rotate(theta):
    theta = np.radians(theta)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]
    )


# apply transformation
def apply_transformation(points, transformation):
    return transformation @ points


# transformation factors
tx, ty = 2, 3
sx, sy = 2, 3
theta = 45

T = translate(tx, ty)
S = scale(sx, sy)
R = rotate(theta)

translated_points = apply_transformation(point_2d, T)
scaled_points = apply_transformation(point_2d, S)
rotated_points = apply_transformation(point_2d, R)

# plotting
plt.figure(figsize=(5,5))
plt.grid()
# original
plt.plot(point_2d[0], point_2d[1], 'ro' ,label='original point (1,1)')
# transformed
plt.plot(translated_points[0], translated_points[1], 'yo', label='translated points')
plt.plot(scaled_points[0], scaled_points[1], 'bo', label='translated points')
plt.plot(rotated_points[0], rotated_points[1], 'go', label='translated points')

plt.plot([point_2d[0][0], translated_points[0][0]], [point_2d[1][0], translated_points[1][0]], 'y--')
plt.plot([point_2d[0][0], scaled_points[0][0]], [point_2d[1][0], scaled_points[1][0]], 'b--')
plt.plot([point_2d[0][0], rotated_points[0][0]], [point_2d[1][0], rotated_points[1][0]], 'g--')

# Labels and plot setup
plt.title('2D Geometric Transformations')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()