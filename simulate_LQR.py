import numpy as np
from qpsolvers import solve_qp
import random 
import math
import scipy
import matplotlib.pyplot as plt

x_dim = 2
u_dim = 2
total_steps = 200

def sim_desired(A, B, Q, R, P, x0, step):
    """
    Uses Euler's discretization to simulate values of x in eqn dx/dt = (A - BK)x
    Calculates using the formula x_i+1 = x_i'*stepsize + x_i
    Stores total_steps number of values of x in the columns of an array 
    returns this array
    """
    x_values = np.zeros((x_dim, total_steps))
    r_values = np.zeros((u_dim, total_steps))

    x_values[:, 0] = x0

    for i in range(1, total_steps):
        r = calculate_r(A,B,Q,R,P, x_values[:, i-1])
        r_values[:, i-1] = r
        x_values[:,i] = step * (np.dot(A, x_values[:, i-1]) + np.dot(B, r)) + x_values[:, i-1]
    return [x_values, r_values]

def calculate_r(A, B, Q, R, P, x):
    """ 
    Given A, B, Q, R, solves for u
    """
    u = np.negative(np.linalg.inv(R)@B.T@P@x)
    return u 
    pass

def make_pos_def(size, max_gen):
    """
    given a maximum value to generate and a size
    returns a random positive definite matrix based on the fact that A^TA is semidefinite if A has independent columns
    """
    A = np.random.rand(size, size)*2*max_gen- max_gen
    out = A.T@A
    return out
    pass

def plot(toplot):
    """
    plots multiple graphs in one chart
    """
    for i in range(toplot.shape[0]):
        plt.plot(toplot[i, 0, :], toplot[i, 1, :], label = i)
    plt.legend()
    plt.show()

def plot3D(toplot):
    """
    plots multiple graphs in one chart
    """
    nums = [None]*total_steps
    for i in range(total_steps):
        nums[i] = i
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(nums, toplot[0, :], toplot[1, :], 'orange')
    plt.show()
    

if __name__ == "__main__":
    #Create random matrices with values between -1 and 1 
    max_gen = 3
    max_2 = 1
    # np.random.seed(0)
    A = np.random.rand(x_dim, x_dim)*2*max_gen- max_gen 
    B = np.random.rand(x_dim, u_dim)*2*max_gen - max_gen*1.5
    Q = make_pos_def(x_dim, 2)
    R = make_pos_def(u_dim, 2)
    x0 = np.random.rand(x_dim,)*2*max_gen - max_gen*1.5
    P = scipy.linalg.solve_continuous_are(A,B,Q,R)

    all_results = sim_desired(A,B, Q, R, P, x0, 0.01)
    x_result = all_results[0]
    u_result = all_results[1]
    print(f'{Q=}')
    print(f'{R=}')
    print(f'{P=}')
    print(f'{x_result=}')
    # print(f'{u_result=}')
    plot3D(x_result)
