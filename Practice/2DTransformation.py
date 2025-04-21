# 2D Transformation
from fnmatch import translate

import numpy as np
import matplotlib.pyplot as plt

# Homogenous points
point_2d = np.array([[1], [1], [1]])

# Translation
def translation(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

# Scaling
def scaling(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

# Rotation
def rotation(theta):
    degree = np.radians(theta)
    return np.array([
        [np.cos(degree), -np.sin(degree), 0],
        [np.sin(degree), np.cos(degree), 0],
        [0, 0, 1]
    ])


def applyTransformation(point, transformation):
    return transformation @ point


tx, ty = 2, 3
sx, sy = 2, 3
theta = 45
T = translation(tx, ty)
S = scaling(sx, sy)
R = rotation(theta)

translatedPoints = applyTransformation(point_2d, T)
scaledPoints = applyTransformation(point_2d, S)
rotatedPoints = applyTransformation(point_2d, R)

print("\nOriginal points: ", point_2d.flatten())
print("Translated points: ", translatedPoints.flatten())
print("Scaled points: ", scaledPoints.flatten())
print("Rotated points: ", rotatedPoints.flatten())

# Plotting
plt.figure(figsize=(5, 5))
plt.grid()

# plotting points
plt.plot(point_2d[0], point_2d[1], 'ro', label=f'Original point (1,1)')
plt.plot(translatedPoints[0], translatedPoints[1], 'bo', label=f'Translated point ({tx}, {ty})')
plt.plot(scaledPoints[0], scaledPoints[1], 'go', label=f'Scaled point ({sx}, {sy})')
plt.plot(rotatedPoints[0], rotatedPoints[1], 'mo', label=f'Rotated point ({theta} degrees)')

# plotting lines between points
plt.plot([point_2d[0][0], translatedPoints[0][0]], [point_2d[1][0], translatedPoints[1][0]], 'b--')
plt.plot([point_2d[0][0], scaledPoints[0][0]], [point_2d[1][0], scaledPoints[1][0]], 'g--')
plt.plot([point_2d[0][0], rotatedPoints[0][0]], [point_2d[1][0], rotatedPoints[1][0]], 'm--')

plt.title('2D Geometric Transformations')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# 2D Transformation using image

import cv2
image  = cv2.imread('F://CV//8bit.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Translation
TX, TY = 100, 50
rows, cols = image.shape[:2]
trans = np.float32([[1,0, TX], [0, 1, TY]])
translatedImg = cv2.warpAffine(image, trans, (cols, rows))

# Scaling
SX, SY = 1.5,2
scaledImg = cv2.resize(image, None, fx=SX, fy=SY, interpolation=cv2.INTER_LINEAR)

# Rotation
angle = 45
center = (cols // 2, rows // 2)
rotates = cv2.getRotationMatrix2D(center, angle, 1)
rotatedImg = cv2.warpAffine(image, rotates, (cols, rows))

# plotting
plt.figure(figsize=(10,5))
plt.subplot(1,4,1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1,4,2)
plt.imshow(translatedImg)
plt.title('Translated Image')

plt.subplot(1,4,3)
plt.imshow(scaledImg)
plt.title('Scaled Image')

plt.subplot(1,4,4)
plt.imshow(rotatedImg)
plt.title('Rotated Image')

plt.tight_layout()
plt.show()
