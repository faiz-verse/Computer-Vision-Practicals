import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- Base Triangle ----------
base_triangle = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [1.5, 2, 0]
])

# ---------- Transform Functions ----------
def translate(points, tx, ty, tz):
    return points + np.array([tx, ty, tz])

def rotate_z(points, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return points @ Rz.T

def scale(points, sx, sy, sz):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz]
    ])
    return points @ S.T

def shear(points, shx=0.5, shy=0.0, shz=0.0):
    shear_matrix = np.array([
        [1, shx, shz],
        [shy, 1, shz],
        [shx, shy, 1]
    ])
    return points @ shear_matrix.T

# ---------- Transformed Versions ----------
original = base_triangle
translated = translate(base_triangle, 1, 1, 1)
rotated = rotate_z(base_triangle, 45)
scaled = scale(base_triangle, 2, 2, 1)
sheared = shear(base_triangle, shx=0.5, shy=0.2)

# ---------- Plot Setup ----------
fig = plt.figure(figsize=(22, 5))
titles = ['Original', 'Translated', 'Rotated', 'Scaled', 'Sheared']
colors = ['blue', 'green', 'red', 'purple', 'orange']
data = [original, translated, rotated, scaled, sheared]

# ---------- Draw Triangle Function ----------
def draw(ax, triangle, title, color):
    ax.set_title(title)
    ax.set_xlim(-3, 4)
    ax.set_ylim(-3, 4)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    poly = Poly3DCollection([triangle], facecolor=color, alpha=0.6)
    ax.add_collection3d(poly)

# ---------- Plot Each in Its Own Axes ----------
for i in range(5):
    ax = fig.add_subplot(1, 5, i + 1, projection='3d')
    draw(ax, data[i], titles[i], colors[i])

plt.tight_layout()
plt.show()
