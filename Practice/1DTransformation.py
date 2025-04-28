import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5])
y = np.array([2,3,5,7,11,13])

plt.plot(x,y, label='Original', marker='o')
plt.title('1D Points')
plt.legend()
plt.grid()
plt.show()

x_trans = x + 2
y_trans = y + 1
plt.plot(x,y, label='Original', marker='o')
plt.plot(x_trans,y_trans, label='Translated', marker='o')
plt.title('Translated Points (x by 2, y by 1)')
plt.legend()
plt.grid()
plt.show()

x_scaled = x * 2
y_scaled = y * 0.5
plt.plot(x,y, label='Original', marker='o')
plt.plot(x_scaled,y, label='X Scaled', marker='o')
plt.plot(x,y_scaled, label='Y Scaled', marker='x')
plt.title('Scaled Points (x by 2, y by 0.5)')
plt.legend()
plt.grid()
plt.show()

