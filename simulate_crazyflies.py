import numpy as np
from qpsolvers import solve_qp
import random 
import math
import scipy
import matplotlib.pyplot as plt

x_dim = 12
u_dim = 12
phi_dim = 12
total_steps = 200

def sim(A, B, Q, R, P, x0, step):
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
        r = calculate_r(B, R, P, x_values[:, i-1])
        r_values[:, i-1] = r
        x_values[:,i] = step * (np.dot(A, x_values[:, i-1]) + np.dot(B, r)) + x_values[:, i-1]
    return [x_values, r_values]

def calculate_r(B, R, P, x):
    """ 
    Given A, B, Q, R, solves for u
    """
    r = np.negative(np.linalg.inv(R)@B.T@P@x)
    return r
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
    graphs a 3D graph
    """
    nums = [None]*total_steps
    for i in range(total_steps):
        nums[i] = i
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(nums, toplot[0, :], toplot[1, :], 'orange')
    plt.show()

def plot3D_multiple(toplot):
    """
    graphs multiple 3D graphs in 1 chart
    """
    toplot_one = toplot[0]
    toplot_two = toplot[1]
    nums = [None]*total_steps
    for i in range(total_steps):
        nums[i] = i
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(nums, toplot_one[0, :], toplot_one[1, :], 'orange')
    ax.plot3D(nums, toplot_two[0, :], toplot_two[1, :], 'purple')
    plt.show()
    pass

def plot2D_multiple(toplot):
    """
    graphs multiple 2D graphs
    """
    fig2, axs2 = plt.subplots(x_dim)
    plt.figure()
    for x in range(x_dim):
        axs2[x].plot(range(0,total_steps), toplot[0][x, :], 'orange') 
        axs2[x].plot(range(0,total_steps), toplot[1][x, :], 'purple')  
        axs2[x].set_title(str(x+1))
    plt.show()
    pass

    
def sim_adaptive_control(A_a, B_a, theta_a, phi_a, init_k_x, init_k_r, init_theta, init_x_0, real_x_val, real_r_val, step):
    """
    Args:
    - A_a: actual A matrix, nxn, unknown
    - B_a: actual B matrix, nxm, known
    - real_x_val: real, desired x values, nxtotal_steps matrix
    - init_k_x: initial k_x matrix, mxn
    - init_k_r: initial k_r matrix, mxm
    - step: number denoting step size of euler approximation

    updates guess for u in the equation x = A_a@x + B_a@(u + f(x))
    returns an array of simulate
    """
    gamma = 0.1

    sim_x_values = np.zeros((x_dim, total_steps))
    sim_x_values[:, 0] = init_x_0
    
    prev_k_x = init_k_x
    prev_k_r = init_k_r
    prev_theta = init_theta

    for i in range(1, total_steps):
        u = prev_k_x@real_x_val[:, i-1] + prev_k_r@real_r_val[:, i-1] - prev_theta@phi_a@real_x_val[:, i-1]
        sim_x_values[:, i] = step*(A_a@sim_x_values[:, i-1] + B_a@(u + theta_a@phi_a@real_x_val[:, i-1])) + real_x_val[:, i-1]
        #calculate change in values: 
        error = real_r_val[:, i] - sim_x_values[:, i]
        change_k_x = - gamma*np.transpose(B_a)@error@np.transpose(real_x_val[:, i])
        change_k_r = - gamma*np.transpose(B_a)@error@np.transpose(real_r_val[:, i])
        change_theta = gamma*np.transpose(B_a)@error@np.transpose(phi_a@real_x_val[:, i])
        #calculate new values:
        prev_k_x = step*change_k_x + prev_k_x
        prev_k_r = step*change_k_r + prev_k_r
        prev_theta = step*change_theta + prev_theta
    
    return sim_x_values



if __name__ == "__main__":
    #represents the DESIRED behaviour with stabilization
    #Create random matrices with values between -maxgen and maxgen
    max_gen = 1
    max_2 = 0.5
    # np.random.seed(0)
    A = np.random.rand(x_dim, x_dim)*2*max_gen- max_gen 
    B = np.random.rand(x_dim, u_dim)*2*max_gen - max_gen*1.5
    Q = make_pos_def(x_dim, 2)
    R = make_pos_def(u_dim, 2)
    x0 = np.random.rand(x_dim,)*2*max_gen - max_gen*1.5
    P = scipy.linalg.solve_continuous_are(A,B,Q,R)

    all_results = sim(A,B, Q, R, P, x0, 0.01)
    x_result = all_results[0]
    r_result = all_results[1]
    print(f'{Q=}')
    print(f'{R=}')
    print(f'{P=}')
    print(f'{x_result=}')
    # print(f'{u_result=}')

    #simulates adaptive control portion:
    A_a = np.random.rand(x_dim, x_dim)*2*max_gen- max_gen #unknown
    B_a = np.random.rand(x_dim, u_dim)*2*max_gen - max_gen*1.5 #known
    theta_a = np.random.rand(u_dim, phi_dim)*2*max_gen- max_gen #unknown
    phi_a = np.random.rand(phi_dim, x_dim)*2*max_gen- max_gen #known
    #we will define phi(x) to be equal to phi_a@x

    #initial guesses: 
    max_guess = 1
    init_k_x = np.random.rand(u_dim, x_dim)*2*max_guess- max_guess
    init_k_r = np.random.rand(u_dim, u_dim)*2*max_guess - max_guess
    init_theta = np.random.rand(u_dim, phi_dim)*2*max_guess- max_guess
    init_x0 = np.random.rand(x_dim,)*2*max_gen - max_gen

    final_sim = sim_adaptive_control(A_a, B_a, theta_a, phi_a, init_k_x, init_k_r, init_theta, init_x0, x_result, r_result, 0.01)
    print(f'{final_sim=}')

    plot2D_multiple([final_sim, x_result])





    

