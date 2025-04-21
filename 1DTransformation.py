# 1D Transformation

import numpy as np
import matplotlib.pyplot as plt

# Original 1D points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11, 13])

plt.plot(x, y, label='Original', marker='o')
plt.title('1D Points')
plt.legend()
plt.grid()
plt.show()

# Translate x to the right by 2 units
x_translated = x + 2

plt.plot(x, y, label='Original', marker='o')
plt.plot(x_translated, y, label='Translated', marker='o')
plt.title('1D Translation')
plt.legend()
plt.grid()
plt.show()

# Scale x and y
x_scaled = x * 2
y_scaled = y * 0.5

plt.plot(x, y, label='Original', marker='o')
plt.plot(x_scaled, y, label='X-Scaled', marker='o')
plt.plot(x, y_scaled, label='Y-Scaled', marker='x')
plt.title('1D Scaling')
plt.legend()
plt.grid()
plt.show()

