# 2D Transformation with image (in case if mam says)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('./Images/8bit.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# 1. Translation (Move image)
tx, ty = 100, 50  # Shift right by 100px, down by 50px
rows, cols = image.shape[:2]
T = np.float32([[1, 0, tx], [0, 1, ty]])
translated_img = cv2.warpAffine(image, T, (cols, rows))

# 2. Scaling (Resize image)
scale_x, scale_y = 1.5, 2
scaled_img = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# 3. Rotation (Around center)
angle = 45
center = (cols // 2, rows // 2) # Getting the center point of the image
R = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv2.warpAffine(image, R, (cols, rows))

# Show all images side-by-side
plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(translated_img)
plt.title('Translated')

plt.subplot(1, 4, 3)
plt.imshow(scaled_img)
plt.title('Scaled')

plt.subplot(1, 4, 4)
plt.imshow(rotated_img)
plt.title('Rotated')

plt.tight_layout()
plt.show()
