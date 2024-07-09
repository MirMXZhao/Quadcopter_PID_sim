"""
Feed in motor speeds to get a trajectory output
"""
import numpy as np
from qpsolvers import solve_qp
import random 
import math
import scipy
import matplotlib.pyplot as plt

# ---------------------------------CONSTANTS---------------------------------
# makes position constants
def make_arm_position():
    """
    makes position column vectors based on the length of the propeller arm
    the front right motor is labelled with 1 and the labelling is done CCW 
    """
    pos_mul = p/math.sqrt(2)
    m1 = pos_mul* np.array([1,1,0]).reshape(-1, 1)
    m2 = pos_mul* np.array([-1,1,0]).reshape(-1, 1) 
    m3 = pos_mul* np.array([-1,-1,0]).reshape(-1, 1)
    m4 = pos_mul* np.array([1,-1,0]).reshape(-1, 1)
    return [m1, m2, m3, m4]

def make_F():
    """
    makes F matrix (multiplied to motor speed matrix)
    """
    forces = np.hstack((cf*e3, cf*e3, cf*e3, cf*e3))
    combine_list = []
    for count, ele in enumerate(positions): 
        combine_list.append(cd*e3*(-1)**count + (cf*np.cross(ele.ravel(), e3.ravel()).reshape(-1,1)))
    torques = np.hstack(tuple(combine_list))
    F = np.vstack((forces, torques))
    return F 

#constants 
m = 29.9 #mass
g = 9.807 #gravity 

cf = 3.1582
cd = 0.0079379

#inertia 
Ixx = 0.001395
Iyy = 0.001395
Izz = 0.002173
J = np.array([[Ixx, 0,0], [0, Iyy, 0], [0, 0, Izz]])

p = 0.03973
e3 = np.array([0,0,1]).reshape(-1, 1)
positions = make_arm_position()
F = make_F()

initial = np.zeros((12, 1))
num_iterations = 5
discretization = 0.01 
desired_motor_speeds = [] 


# ---------------------------------ACTUAL SIMULATION--------------------------------
def make_state():
    """
    motor_speeds is a list of 4 by 1 column vectors describing the speeds of the 
    four motors.
    state vector is a 12 by 1 vector representing p, v, angles, omega in that order
    """
    store_state = simulate_runge_kutta(state_func, initial, 0, 2)
    return store_state

def make_omega(v):
    """
    makes the omega vector given vector v representing the four motor speeds
    """
    output = v*np.absolute(v)
    return output

def calculate_R(psi, theta, phi):
   """
   - creates rotation matrices
   - three angles represent yaw, pitch, roll respectively
   """
   yaw = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0,0,1]])
   pitch = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta),0,math.cos(theta)]])
   roll = np.array([[1, 0, 0],[0, math.cos(phi), -math.sin(phi)],[0, math.sin(phi), math.cos(phi)]])
   R = yaw@pitch@roll
   return R

def state_func(x, t):
   """
   math to calculate derivative of state
   """
   p_dot = x[3:6]
#    print(f'{p_dot = }')
   # makes the new acceleration and angular acceleration vectors
   cur_motors_abs = make_omega(motor_speeds[int(t)]) 
   to_extract = np.vstack((-m*g*e3, -(np.cross(x[-3:].reshape(1,-1), (J@x[-3:]).reshape(1, -1))).reshape(-1, 1)))
   R = calculate_R(x[6], x[7], x[8])
#    print(f'{R = }')
   print(f'{F@cur_motors_abs = }')
   sub_step = np.block([[R, np.zeros((3,3))],[np.zeros((3,3)), np.eye(3)]])
   to_extract = to_extract + sub_step@F@cur_motors_abs
   v_dot = to_extract[:3]/m
#    print(f'{v_dot = }')
   omega_dot = np.linalg.inv(J)@to_extract[-3:]
#    print(f'{omega_dot = }')
   angle_dot = make_M_inv(x[6], x[7], x[8])@x[-3:]
#    print(f'{angle_dot = }')
   output = np.vstack((p_dot, v_dot, angle_dot, omega_dot))
   return output

def make_M_inv(phi, theta, omega):
    """
    - makes the inverse of the M matrix, for calculating angle_dot
    """
    M = np.array([[1, 0, -math.sin(theta)], 
                  [0, math.cos(phi), math.cos(theta)*math.sin(phi)], 
                  [0, -math.sin(phi), math.cos(theta)*math.cos(phi)]])
    return np.linalg.inv(M)

def simulate_runge_kutta(fn, x0, t0, step_size):
   """
   - RK4
   """
   simulated_x = [None]*num_iterations
   simulated_x[0] = x0
   cur_t =t0
   for i in range(1, num_iterations):
       k1 = fn(simulated_x[i-1], cur_t)
       k2 = fn(simulated_x[i-1] + step_size*k1/2, cur_t + step_size/2)
       k3 = fn(simulated_x[i-1] + step_size*k2/2, cur_t + step_size/2)
       k4 = fn(simulated_x[i-1] + step_size*k3, cur_t + step_size)
       simulated_x[i] = simulated_x[i-1] + step_size/6*(k1 + 2*k2 + 2*k3 + k4)
       cur_t = cur_t + step_size
   return simulated_x

# ---------------------------------TESTING--------------------------------
def make_motor_speeds():
    """
    - returns a list of 2*num_iterations motor speeds
    - motor speeds are a column vector of size 4
    """
    throttle_control(40, num_iterations)
    yaw_control(True, num_iterations)

def throttle_control(mag, reps):
    """
    - up: A boolean. if True, want to go up. Otherwise down.
    - mag: a float representing the strength of the control
    """
    global desired_motor_speeds

    for i in range(reps):
        new_thrust = np.array([[1],[-1],[1],[-1]])*mag
        desired_motor_speeds.append(new_thrust)

def yaw_control(left, reps):
    """
    - up: A boolean. if True, want to go up. Otherwise down.
    - mag: a float representing the strength of the control
    """
    global desired_motor_speeds

    for i in range(min(reps, 10)):
        if left: 
            new_yaw = np.array([[3], [-1], [1], [-3]])*10
        else:
            new_yaw = np.array([[1], [-3], [3], [-1]])*10
        desired_motor_speeds.append(new_yaw)
    pass

if __name__ == "__main__":
    #stuff
    print(f'{positions = }')
    print(f'{F = }')
    
    #testing:
    make_motor_speeds()
    print(f'{motor_speeds = }')
    print(make_state()) 

    pass