import numpy as np
from pycrazyswarm import *
from qpsolvers import solve_qp
import random 
import math
import scipy
import matplotlib.pyplot as plt

x_dim = 2
u_dim = 1
phi_dim = 1
total_steps = 200
sleepRate = 20

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs
num_cf = len(allcfs.crazyflies)
startPos = [None]*num_cf

def goalMotion(totalTime, radius, time_i):
    startTime = timeHelper.time()
    pos = [None]*num_cf
    center_circle = [None]*num_cf
    goal = np.zeros((2*num_cf, 3))

    for i in range(num_cf):
        pos[i] = allcfs.crazyflies[i].position()
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    # time = timeHelper.time() - startTime
    speed = 0.2 #makes it go slower
    time = time_i*speed
    for i in range(num_cf):
        omega = 0.5* np.pi / totalTime[i]
        vx = -radius[i] * omega * np.sin(omega * (time + speed))  
        vy = radius[i] * omega * np.cos(omega * (time + speed))
        velocity = np.array([vx, vy, 0])
        position = center_circle[i] + radius[i] * np.array([np.cos(omega * time), np.sin(omega * time), 0])
        acceleration = 0.1*(position - center_circle[i])
        yaw = 0.1
        omega = np.array([0.1, 0 ,0])
        allcfs.crazyflies[i].cmdFullState(position, velocity, acceleration, yaw, omega)
        timeHelper.sleepForRate(sleepRate)

    return goal #goal encodes desired position in odd indices and desired velocity in even indices

def sim_non_stable(A, B, g, r, x0, step):
    """
    Uses Euler's discretization to simulate values of x in eqn dx/dt = (A - BK)x
    Calculates using the formula x_i+1 = x_i'*stepsize + x_i
    Stores total_steps number of values of x in the columns of an array 
    returns this array
    """
    x_values = np.zeros((x_dim, total_steps))
    r_values = np.zeros((u_dim, total_steps))

    x_values[:, 0] = x0[:, 0]
    for i in range(1, total_steps):
        # print(np.dot(A, x_values[:, i-1]))
        # print(B*phi(x_values[:, i-1], g))
        x_values[:,i] = step * (np.dot(A, x_values[:, i-1]) - B[:, 0]*phi(x_values[:, i-1], g)*1/r) + x_values[:, i-1]
        r_values[:, i-1] = phi(x_values[:, i-1], g)*1/r

    return [x_values, r_values]

def phi(x, g):
    """
    - x is a 2 by 1 array with the first value representing theta
    - g is the gravitational constant
    """
    theta = x[0]
    gravity_force = g*math.sin(theta)
    # print(gravity_force)
    return gravity_force

def plot(toplot):
    """
    - toplot: list of arrays
    plots multiple graphs in one chart
    """
    for i in range(len(toplot)):
        plt.plot(range(0,total_steps), toplot[i], label = i)
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
    fig2, axs2 = plt.subplots(2)
    plt.figure()
    axs2[0].plot(range(0,total_steps), toplot[0], 'orange') 
    axs2[0].set_title("theta")
    axs2[1].plot(range(0,total_steps), toplot[1], 'purple')  
    axs2[1].set_title("omega")
    plt.show()
    pass

    
def sim_adaptive_control(A_a, B_a, g_a, lambda_a, init_k_x, init_k_r, init_theta, init_x_0, real_x_val, real_r_val, step):
    """
    Args:
    - A_a: actual A matrix, nxn, unknown
    - B_a: actual B matrix, nxm, known
    - real_x_val: real, desired x values, nxtotal_steps matrix
    - init_k_x: initial k_x matrix, mxn
    - init_k_r: initial k_r matrix, mxm
    - step: number denoting step size of euler approximation

    updates guess for u in the equation x_dot = A_a@x + B_a@(u + f(x))
    A_ax + B_alambda(u+theta phi)
    returns an array of simulated values
    """
    gamma = 0.1

    sim_x_values = np.zeros((x_dim, total_steps))
    sim_x_values[:, 0] = init_x_0
    
    prev_k_x = init_k_x
    prev_k_r = init_k_r
    prev_theta = init_theta

    for i in range(1, total_steps):
        # print(f'{np.shape(prev_k_x)=}')
        # print(f'{np.shape(real_x_val[:, i-1])=}')
        # print(f'{np.shape(prev_k_r)=}')
        # print(f'{np.shape(real_r_val[:, i-1])=}')
        # print(f'{np.shape(prev_theta)=}')
        # print(f'{np.shape(phi(real_x_val[:, i-1], g_a))=}')
        u = prev_k_x@real_x_val[:, i-1] + prev_k_r@real_r_val[:, i-1] - prev_theta@phi(real_x_val[:, i-1], g_a)
        # print(f'{A_a@sim_x_values[:, i-1]=}')
        # print(f'{B_a@(u + g_a@phi(real_x_val[:, i-1], g_a))=}')
        sim_x_values[:, i] = step*(A_a@sim_x_values[:, i-1] + (B_a@(u + g_a@phi(real_x_val[:, i-1], g_a))).T) + real_x_val[:, i-1]
        #calculate change in values: 
        error = real_r_val[:, i] - sim_x_values[:, i]
        error = error[np.newaxis, :]
        error = np.transpose(error)
        x_val = np.reshape(real_x_val[:, i], (real_x_val[:, i].size, 1))
        # print(f'{np.shape(B_a)=}')
        # print(f'{np.shape(error)=}')
        # print(f'{np.shape(x_val)=}')
        change_k_x = - gamma*np.transpose(B_a)@error@np.transpose(x_val)
        change_k_r = - gamma*np.transpose(B_a)@error@np.transpose(real_r_val[:, i])
        change_theta = gamma*np.transpose(B_a)@error@phi(real_x_val[:, i-1], g_a)
        #calculate new values:
        prev_k_x = step*change_k_x + prev_k_x
        prev_k_r = step*change_k_r + prev_k_r
        prev_theta = step*change_theta + prev_theta
    
    return sim_x_values
    pass


if __name__ == "__main__":
    goalMotion([4,4,4,4][1,1,1,1] )