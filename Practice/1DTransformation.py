import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,1,2,3,4,5])
y = np.array([2,3,5,7,11,13])

plt.plot(x,y, label='original points', marker='o')
plt.title('Initial 1D Points')
plt.legend()
plt.grid()
plt.show()

# Translation
xt = x + 2
plt.plot(x,y, label='original points', marker='o')
plt.plot(xt,y, label='translated points', marker='x')
plt.title('1D Translation')
plt.legend()
plt.grid()
plt.show()

# Scaling
xs = x * 2
ys = y * 0.5
plt.plot(x,y, label='original points', marker='o')
plt.plot(xs,y, label='x scaled points', marker='x')
plt.plot(x,ys, label='y scaled points', marker='x')
plt.title('1D Scaling')
plt.legend()
plt.grid()
plt.show()


