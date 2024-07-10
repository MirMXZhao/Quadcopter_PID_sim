import numpy as np
from qpsolvers import solve_qp
import random
import math
import scipy
import matplotlib.pyplot as plt


#############################################################################
# ---------------------------------CONSTANTS--------------------------------#
#############################################################################


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


#number of iterations
num_iterations = 2000

#constants
m = 29.9 #mass
g = 9.807 #gravity


#desired control constants
cf = 3.1582
cd = 0.0079379


# #real constants #unnecessary
# t_cf = cf + random.choice([random.uniform(cf/7, cd/2), random.uniform(-cf/2, -cf/7)])
# t_cd = cd + random.choice([random.uniform(cd/7, cd/2), random.uniform(-cd/2, -cd/7)])


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

real_initial = np.zeros((12,1))
# real_initial = np.random.uniform(low = -1, high = 2, size = (12,1))
real_initial[2] = 20
real_initial[0] = 0
real_initial[1] = 0

#PID Parameters 
kp = 400
kv = 150
kr = 10
kw = 1

#DISCRETIZATION
discretization = 0.003

###############################################################################
# ---------------------------------MOTOR SPEEDS-------------------------------#
###############################################################################

#Motor controls for testing
#quadrant 1 is motor 1, quadrant 2 is motor 4, quadrant 3 is motor 3, quadrant 4 is motor 2
throttle_up = np.array([[1],[1],[1],[1]])
roll_left = np.array([[1], [1], [1.02], [1.02]])
roll_right = np.array([[1.02], [1.02], [1], [1]])
pitch_forward = np.array([[1], [1.02], [1.02], [1]])
pitch_backward = np.array([[1.02], [1], [1], [1.02]])
yaw_CW = np.array([[1], [1.1], [1], [1.1]])
yaw_CCW = np.array([[1.1], [1], [1.1], [1]])

def make_motor_speeds_non_int():
   stored = []
   stored.append((0, throttle_up*0))
   stored.append((0.1, throttle_up*6))
   stored.append((1, throttle_up*7))
   stored.append((1.05, pitch_forward*1.4))
   stored.append((1.1, throttle_up *3))
   stored.append((2.2, throttle_up*3))
   stored.append((2.4, pitch_backward*1.3))
   stored.append((3, throttle_up*3))
   stored.append((20, throttle_up*3))
   return stored 

new_desired_motor_speeds = make_motor_speeds_non_int()

index = 0

def cts_motor_speed_fn(t):
   global index
   if t >= new_desired_motor_speeds[index+1][0]:
      index +=1 
   fn_output = new_desired_motor_speeds[index][1] + (t - new_desired_motor_speeds[index][0])/(new_desired_motor_speeds[index+1][0] - new_desired_motor_speeds[index][0])*(new_desired_motor_speeds[index + 1][1] - new_desired_motor_speeds[index][1])
   # print(fn_output)
   return fn_output

#######################################iwi####################################
# ---------------------------------MAKE STATES-------------------------------#
##############################################################################
# makes num_iterations state vectors
# each state vector is a 12 by 1 vector representing p, v, angles, omega in that order

def make_state():
   """
   motor_speeds is a list of 4 by 1 column vectors describing the speeds of the
   four motors.
   state vector is a 12 by 1 vector representing p, v, angles, omega in that order
   """
   store_state = simulate_runge_kutta(state_func, initial, 0, discretization, num_iterations, True)
   return store_state

def make_omega(v):
   """
   makes the omega vector given vector v representing the four motor speeds
   """
   output = v*np.absolute(v)
   return output

def calculate_R(phi, theta, psi):
  """
  - creates rotation matrices
  - three angles represent roll, pitch, yaw respectively
  """
  yaw = np.array([[math.cos(psi), -math.sin(psi), 0], 
                  [math.sin(psi), math.cos(psi), 0], 
                  [0, 0, 1]])
  pitch = np.array([[math.cos(theta), 0, math.sin(theta)], 
                    [0, 1, 0], 
                    [-math.sin(theta),0,math.cos(theta)]])
  roll = np.array([[1, 0, 0],
                   [0, math.cos(phi), -math.sin(phi)],
                   [0, math.sin(phi), math.cos(phi)]])
  R = yaw @ pitch @ roll
  return R

def state_func(x, t):
  """
  math to calculate derivative of state
  """
  p_dot = x[3:6]
  # makes the new acceleration and angular acceleration vectors
  cur_motors_abs = make_omega(cts_motor_speed_fn(t))
  to_extract = np.vstack((-m*g*e3, -(np.cross(x[-3:].reshape(1,-1), (J@x[-3:]).reshape(1, -1))).reshape(-1, 1)))
  R = calculate_R(x[6], x[7], x[8])
#    print(f'{F@cur_motors_abs = }')
  sub_step = np.block([[R, np.zeros((3,3))],[np.zeros((3,3)), np.eye(3)]])
  to_extract = to_extract + sub_step@F@cur_motors_abs
#    print(f'{to_extract = }')
  v_dot = to_extract[:3]/m
  omega_dot = np.linalg.inv(J)@to_extract[-3:]
  angle_dot = make_M_inv(x[6], x[7], x[8])@x[-3:]
  output = np.vstack((p_dot, v_dot, angle_dot, omega_dot))
  return output

def make_M_inv(phi, theta, psi):
   """
   - makes the inverse of the M matrix, for calculating angle_dot
   """
   M = np.array([[1, 0, -math.sin(theta)],
                 [0, math.cos(phi), math.cos(theta)*math.sin(phi)],
                 [0, -math.sin(phi), math.cos(theta)*math.cos(phi)]])
   return np.linalg.inv(M)

des_lin_accel = [np.array([[0], [0], [0]])]
des_rot_accel = [np.array([[0], [0], [0]])]

def simulate_runge_kutta(fn, x0, t0, step_size, num_iters, store = False):
  """
  - RK4
  #used in the calculation of desired states 
  #as well as the calculation of the real states 
  """
  #Something about step_size is weird
  global des_lin_accel
  global des_rot_accel

  simulated_x = [None]*num_iters
  simulated_x[0] = x0
  cur_t =t0
  for i in range(1, num_iters):
      # print(f'{simulated_x[i-1] = }')
      k1 = fn(simulated_x[i-1], cur_t)
      # use STORE VARIABLE TO store the values from k1 
      k2 = fn(simulated_x[i-1] + step_size*k1/2, cur_t + step_size/2)
      k3 = fn(simulated_x[i-1] + step_size*k2/2, cur_t + step_size/2)
      k4 = fn(simulated_x[i-1] + step_size*k3, cur_t + step_size)
      # print(f'{k1 = }')
      # print(f'{k2 = }')
      # print(f'{k3 = }')
      # print(f'{k4 = }')
      new_slope = (k1 + 2*k2 + 2*k3 + k4)/6
      if store:
         des_lin_accel.append(new_slope[3:6])
         des_rot_accel.append(new_slope[-3:])
      simulated_x[i] = simulated_x[i-1] + step_size*new_slope
      cur_t = cur_t + step_size
  return simulated_x

# print(f'{des_lin_accel =}')
desired_states = make_state() # this is the array of desired states
# print(f'{desired_states = }')

################################################################################
# ---------------------------------PID CONTROLLER------------------------------#
################################################################################
created_motor_speeds = []

def calculate_x_tilde(yaw):
   x_tilde = np.array([[math.cos(yaw)], [math.sin(yaw)], [0]])
   return x_tilde

def calculate_Rwd(val, x_tilde):
   """
   val is -kp*ep -kv*ev + mge3 + mpwd
   this is a 3 by 1 column vector
   returns the desired rotational matrix
   """
   zwd = (val)/(np.linalg.norm(val))
   second_column = (np.cross(zwd.ravel(), x_tilde.ravel())/(np.linalg.norm(np.cross(zwd.ravel(), x_tilde.ravel())))).reshape(3,1)
   first_column = (np.cross(second_column.ravel(), zwd.ravel())/(np.linalg.norm(np.cross(second_column.ravel(), zwd.ravel())))).reshape(3,1)

   Rwd = np.hstack((first_column, second_column, zwd))
   return Rwd

def bound(val1, val2, bound):
   """
   val1 and val2 are two column vectors of size 3
   ensures that each row in val1 is within val2 +- bound. 
   not currently in use
   """
   new_list = []
   for i in range(3):
      new_list.append(max(val2[i] - bound, min(val1[i], val2[i] + bound)))
   new_vec = np.array(new_list).reshape(-1, 1)
   return new_vec

def PID(cur_real, t):
   """
   PID controller: calculates next force and torque
   Only for one iteration
   """
   iter = int(t/discretization)

   cur_desired = desired_states[iter]

   # position and velocity errors
   ep =  cur_real[:3] - cur_desired[:3] 
   ev = cur_real[3:6] - cur_desired[3:6]

   x_tilde = calculate_x_tilde(cur_desired[8]) 
   val = -kp*ep - kv*ev + m*g*e3 + m*des_lin_accel[iter]
   rwd = calculate_Rwd(val, x_tilde)
   rwb = calculate_R(cur_real[6], cur_real[7], cur_real[8])
   temp_matrix = rwd.T@rwb -rwb.T@rwd
   fBz = np.dot(val.reshape(3,), (rwb@e3).reshape(3,))
   
   #angle and rotational velocity errors  
   er = 0.5*np.array([[temp_matrix[2][1]], [temp_matrix[0][2]], [temp_matrix[1][0]]])
   ew = cur_real[-3:] - rwb.T@rwd@cur_desired[-3:]

   ax = cur_real[9].item()
   ay = cur_real[10].item()
   az = cur_real[11].item()
   cur_real_cross = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
   torqueB = -kr*er -kw*ew + np.cross(cur_real[-3:].ravel(), (J@cur_real[-3:]).ravel()).reshape(3,1) - J@(cur_real_cross@rwb.T@rwd@cur_desired[-3:] - rwb.T@rwd@des_rot_accel[iter])
   ang_accel = np.linalg.inv(J)@(torqueB - (np.cross(cur_real[-3:].ravel(), (J@cur_real[-3:]).ravel())).reshape(3,1))
   final_ang = make_M_inv(cur_real[6], cur_real[7], cur_real[8])@(ang_accel)
   final = np.vstack((cur_real[3:6], (rwb@(fBz*e3)- m*g*e3)/m, cur_real[-3:], final_ang))

   return final

created_states = simulate_runge_kutta(PID, real_initial, 0, discretization, num_iterations-2)
# print(f'{created_states = }')

###########################################################################
# --------------------------EVALUATION METRIC-----------------------------#
###########################################################################

def evaluate_performance():
   #average distance between real pos and desired pos
   total_sum = 0
   for i in range(num_iterations-2):
      total_sum = total_sum + np.linalg.norm(created_states[i][:3] - desired_states[i][:3])
   return total_sum/(num_iterations-2)

###########################################################################
# ---------------------------------PLOTTING-------------------------------#
###########################################################################
def plot_1axis(toplot):
   """
   plots multiple graphs in one chart
   """
   labels = ["pos1", "pos2", "pos3", "vel1", "vel2", "vel3", "ang1", "ang2", "ang3", "ome1", "ome2", "ome3"]
   x_vals = range(num_iterations)
   for i in range(np.shape(toplot[1])[0]):
       y_vals = [vector[i] for vector in toplot]
       plt.plot(x_vals, y_vals, label = labels[i])
   plt.legend()
   plt.show()

def plot_mulaxis(toplot):
   """
   plots multiple graphs in many axes
   """
   fig, axs = plt.subplots(4, 3)
   labels = ["pos1", "pos2", "pos3", "vel1", "vel2", "vel3", "ang1", "ang2", "ang3", "ome1", "ome2", "ome3"]
   x_vals = range(num_iterations)
   for i in range(12):
       y_vals = [vector[i] for vector in toplot]
       axis_1 = math.floor(i/3)
       axis_2 = i%3
       axs[axis_1, axis_2].plot(x_vals, y_vals)
       axs[axis_1, axis_2].set_title(labels[i])
   plt.legend()

def plot2D_both(toplot1, toplot2):
   """
   plots multiple graphs in many axes
   """
   fig, axs = plt.subplots(4, 3, layout = "constrained")
   labels = ["pos1", "pos2", "pos3", "vel1", "vel2", "vel3", "ang1", "ang2", "ang3", "ome1", "ome2", "ome3"]
   x_vals1 = []
   x_vals2 = []
   for i in range(0, len(toplot1)):
      x_vals1.append(i*discretization)
   for i in range(0, len(toplot2)):
      x_vals2.append(i*discretization)

   for i in range(12):
       y_vals1 = [vector[i] for vector in toplot1]
       y_vals2 = [vector[i] for vector in toplot2]
       axis_1 = math.floor(i/3)
       axis_2 = i%3
       axs[axis_1, axis_2].plot(x_vals1, y_vals1, label = "created")
       axs[axis_1, axis_2].plot(x_vals2, y_vals2, label = "desired")
       axs[axis_1, axis_2].set_title(labels[i])
   plt.legend()


def plot3D(toplot):
   """
   graphs a 3D graph
   """
   fig = plt.figure()
   ax = plt.axes(projection='3d')
   x_val = [vector[0] for vector in toplot]
   y_val = [vector[1] for vector in toplot]
   z_val = [vector[2] for vector in toplot]
   ax.plot3D(x_val, y_val, z_val, 'orange')

def plot3D_both(toplot1, toplot2):
   """
   plot both
   """
   fig = plt.figure()
   ax = plt.axes(projection='3d')
   x_val1 = [vector[0] for vector in toplot1]
   y_val1 = [vector[1] for vector in toplot1]
   z_val1 = [vector[2] for vector in toplot1]
   x_val2 = [vector[0] for vector in toplot2]
   y_val2 = [vector[1] for vector in toplot2]
   z_val2 = [vector[2] for vector in toplot2]
   ax.plot3D(x_val1, y_val1, z_val1, 'blue', label = "real")
   ax.plot3D(x_val2, y_val2, z_val2, 'orange', label = "desired")

if __name__ == "__main__":

   #stuff
   # print(f'{evaluate_performance() = }')

    #testing:
    # plot_1axis(state)
    # plot_mulaxis(state)
    # plot3D(state)

    #testing PID
    # plot_mulaxis(created_states, num_iterations-1)
    # plot_mulaxis(desired_states)
   #  print(f'{created_states = }')
   #  print(f'{desired_states = }')
    print(f'{evaluate_performance() = }')
    plot2D_both(created_states, desired_states)
    # plot3D(created_states)
    plot3D_both(created_states, desired_states)
    plt.show()

