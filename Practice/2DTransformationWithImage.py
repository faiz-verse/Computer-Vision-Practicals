from idlelib.pyparse import trans

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../Images/8bit.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tx, ty = 100, 50
rows, cols = image.shape[:2]
T = np.float32([[1, 0, tx], [0,1,ty]])
translated_img = cv2.warpAffine(image, T, (cols,rows))

sx, sy = 2, 1.5
scaled_img = cv2.resize(image, None ,fx= sx, fy=sy, interpolation= cv2.INTER_LINEAR)

theta = 45
center = (cols//2, rows//2)
R = cv2.getRotationMatrix2D(center, theta, 1)
rotated_img = cv2.warpAffine(image, R, (cols,rows))

plt.figure(figsize=(10,5))
plt.subplot(1,4,1)
plt.imshow(image)
plt.title('Original image')

plt.subplot(1,4,2)
plt.imshow(translated_img)
plt.title('Translated image')

plt.subplot(1,4,3)
plt.imshow(scaled_img)
plt.title('Scaled image')

plt.subplot(1,4,4)
plt.imshow(rotated_img)
plt.title('Rotated image')

plt.show()

#
# cv2.imshow("Translated image", translated_img)
# cv2.imshow("Scaled image", scaled_img)
# cv2.imshow("Rotated image", rotated_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()