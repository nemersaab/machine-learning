import numpy as np
import matplotlib.pyplot as plt
import math

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
def gradients(x,y,w,b):
    m=x.shape[0]
    djw=0
    djb=0
    for i in range(m):
        f_wb = w*x[i]+b
        djw +=(f_wb - y[i])*x[i]
        djb += f_wb -y[i]
    return djw/m ,djb/m

def gradient_descent(x,y,w_in,b_in ,alpha,number_of_iterations):
    w=w_in
    b=b_in

    j_history=[]
    p_history=[]

    for i in  range(number_of_iterations):
        djw , djb = gradients(x,y,w,b)
        w= w - alpha * djw
        b= b - alpha * djb

        if(i<100000):
            j_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])

        if(i%math.ceil(number_of_iterations/10)==0):
            print(f"in iteration:{i} w={w} b={b} cost={cost_function(x,y,w,b)} ")    
    return w,b,j_history,p_history    


x_train = np.array([1.5, 2,1.2,1.8,2.5]) 
y_train = np.array([350,450,280,390,600])

w,b,jhist,phist=gradient_descent(x_train,y_train,0,0,0.01,10000)
whist = []
bhist = []
for i in range(len(phist)):
    whist.append(phist[i][0])
    bhist.append(phist[i][1])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(whist,bhist,jhist)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('cost function')
plt.show()