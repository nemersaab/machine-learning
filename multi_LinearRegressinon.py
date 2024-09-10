import numpy as np
import math
import matplotlib.pyplot as plt
def single_itiration_value(w,x,b):
    f=0
    n=x.shape[0]
    for i in range(n):
        f+=w[i]*x[i]
    return f+b  
def single_itiration_val(w,x,b):
    return np.dot(w,x)+b
def cost_function(x,y,w,b):
    m=x.shape[0]
    cost = 0
    for i in range(m):
        cost+=(single_itiration_val(w,x[i],b)-y[i])**2
    return cost /(2*m)    
def z_score_normalization(X):
    mu = np.mean(X,0)
    sigma = np.std(X,0)
    X_normailzed = (X-mu)/sigma
    return mu,sigma,X_normailzed

def gradients(X:np.ndarray,y:np.ndarray,w:np.ndarray,b:float):
    m,n=X.shape
    djw=np.zeros(n)
    djb=0
    for i in range(m):
        err = np.dot(X[i],w)+b - y[i]
        djw+=err * X[i]
        djb+=err
    return djw /m , djb/m        
def gradient_descent(X,y,w_in,b_in,alpha,nbr_iterations):
    b=b_in
    w=[]
    jhist=[]
    for i in range(len(w_in)):
        w.append(w_in[i])
    for i in range(nbr_iterations):
        djw,djb=gradients(X,y,w,b)
        
        w-= alpha*djw
        b-= alpha*djb 

        if(i<10000):
            jhist.append(cost_function(X,y,w,b))

        if i% math.ceil(nbr_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {jhist[-1]:8.2f}   ")
        
    return w, b, jhist 
x_train=np.random.random_sample(15).reshape(3,5)
y_train = np.random.random_sample(3)
w,b,jhist = gradient_descent(x_train,y_train,[0,0,0,0,0],0,0.01,10000)

mu,sigma,X_normalized=z_score_normalization(x_train)
w_normal,b_normal,jhist_normal=gradient_descent(X_normalized,y_train,[0,0,0,0,0],0,0.01,10000)

y_pred_unormal = np.dot(x_train,w)+b
y_pred_normal = np.dot(X_normalized,w_normal)+b_normal

