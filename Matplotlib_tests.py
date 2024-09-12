import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the cost function
def cost_function(x, y, w, b):
    m = x.shape[0] 
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# Define ranges for w and b
w = np.linspace(-100, 100, 400)  # 100 points in range -10 to 10
b = np.linspace(-100, 100, 400)

# Training data
x_train = np.array([1, 2,1,3,2]) 
y_train = np.array([1.5,1.5,1,4,2.5])

# Initialize a 2D array to store cost values
J = np.zeros((w.size, b.size))

# Calculate cost for every combination of w and b
for i in range(w.size):
    for j in range(b.size):
        J[i, j] = cost_function(x_train, y_train, w[i], b[j])

# Create meshgrid for surface plot
W, B = np.meshgrid(w, b)

# Plotting the surface
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(W, B, J.T, cmap='viridis')  # Transpose J to match dimensions
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost')
plt.show()
