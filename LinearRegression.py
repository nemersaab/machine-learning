import numpy as np
import matplotlib.pyplot as plt



x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

plt.plot(x_train,y_train,marker='x',c='r')
plt.title("test")
plt.show()

def pridictions(x,w,b):
    m=x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i] + b
    return f_wb
def cost_function(x,y,w,b):
    m=x.shape[0] 
    cost_sum = 0
    for i in range(m):
        f_wb = w*x[i]+b
        cost = (f_wb -y[i])**2
        cost_sum += cost
    total_cost = (1/(2*m))*cost_sum
    return total_cost
