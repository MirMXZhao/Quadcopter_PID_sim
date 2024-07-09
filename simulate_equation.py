import numpy as np
from qpsolvers import solve_qp
import random 
import math
import matplotlib.pyplot as plt

x_dim = 2
u_dim = 2
total_steps = 30

def sim(A, B, K, x0, step):
    """
    Uses Euler's discretization to simulate values of x in eqn dx/dt = (A - BK)x
    Calculates using the formula x_i+1 = x_i'*stepsize + x_i
    Stores total_steps number of values of x in the columns of an array 
    returns this array
    """
    x_values = np.zeros((x_dim, total_steps))

    x_values[:, 0] = x0
    for i in range(1, total_steps):
        x_values[:,i] = step * (np.dot(A, x_values[:, i-1]) - np.dot(B, np.dot(K, x_values[:, i-1]))) + x_values[:, i-1]
    return x_values

def plot(toplot):
    """
    plots multiple graphs in one chart
    """
    for i in range(toplot.shape[0]):
        plt.plot(toplot[i, 0, :], toplot[i, 1, :], label = i)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    #Create random matrices with values between -1 and 1 
    max_gen = 3
    A = np.random.rand(x_dim, x_dim)*2*max_gen- max_gen 
    B = np.random.rand(x_dim, u_dim)*2*max_gen - max_gen*1.5
    x0 = np.random.rand(x_dim,)*2*max_gen - max_gen*1.5
    K = np.random.rand(u_dim, x_dim)*2*max_gen - max_gen*1.5

    print(A)
    print(B)
    print(x0)

    simmed_val_s = sim(A, B, K, x0, 0.01)
    # simmed_val_m = sim(A, B, K, x0, 0.1)
    # simmed_val_b = sim(A, B, K, x0, 0.3)
    # alldata = np.array([simmed_val_s, simmed_val_m])
   # alldata = np.array([simmed_val_s, simmed_val_m, simmed_val_b])
    alldata = np.array([simmed_val_s])
    # print(alldata.shape)
    # print(alldata)
    print(simmed_val_s)
    # print(simmed_val_m) 
    # print(simmed_val_b)
    plot(alldata)

    #supposed to have complex roots but i think i did the math wrong :((
    A1 = np.array([[3, 10],[-3, -3]])
    B1 = np.array([1, -2]).T
    K1 = np.array([2, 3])
    x0= np.array([0.1,0.1])
    simmed_val = sim(A1, B1, K1, x0, 0.01)
    print(simmed_val)
    alldata2 = np.array([simmed_val])
    plot(alldata2)
    # np.linalg.pinv()
    

