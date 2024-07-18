"""
- Creates desired motor speeds
- Simulates quadcopter motion based on the desired motor speed 
  > creates an array of desired states
- The process above can be replaced with manual input of the states 
- The real quadcopter (with different initial states) follows the desired speeds using PID
  > creates an array of created states (converges to desired state
"""
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
   makes F matrix (multiplied to motor speed matrix) to calculate desired states
   """
   forces = np.hstack((cf*e3, cf*e3, cf*e3, cf*e3))
   combine_list = []
   for count, ele in enumerate(positions):
       combine_list.append(cd*e3*(-1)**count + (cf*np.cross(ele.ravel(), e3.ravel()).reshape(-1,1)))
   torques = np.hstack(tuple(combine_list))
   F = np.vstack((forces, torques))
   return F

#NUMBER OF ITERATIONS 
num_iterations = 2000

#DISCRETIZATION
step = 0.003

#constants
m = 29.9 #mass
g = 9.807 #gravity

#desired control constants
cf = 3.1582
cd = 0.0079379

#inertia
Ixx = 0.001395
Iyy = 0.001395
Izz = 0.002173
J = np.array([[Ixx, 0,0], [0, Iyy, 0], [0, 0, Izz]])

#arm positions
p = 0.03973
positions = make_arm_position()

#constants needed for calculation
e3 = np.array([0,0,1]).reshape(-1, 1)
F = make_F()

#desired starting state
des_initial = np.zeros((12, 1))

#starting state of the real drone
real_initial = np.zeros((12,1))
real_initial[0] = 1
real_initial[1] = 0
real_initial[2] = 0

#global variables changed later on:
desired_motor_speeds = [0]
#PID Parameters 
kp = 400
kv = 150
kr = 10
kw = 1

#Adaptive control parameters
gamma = 0.15
# A = np.zeros((4,4))
# A[0,2] = 1
# A[1,3] = 1
#initial conditions for kx, kr, thet 
kx = np.identity(4)*0.1
thet= np.identity(4)*0.1

###############################################################################
# -------------------------------------RK4------------------------------------#
###############################################################################

def simulate_runge_kutta(fn, x0, t0, step_size, num_iters, des, store = False):
  """
  - RK4
  - also stores the desired linear and rotational accelerations 
  """
  global des_lin_accel
  global des_rot_accel

  simulated_x = [None]*num_iters
  simulated_x[0] = x0
  cur_t =t0
  for i in range(1, num_iters):
      k1 = fn(des, simulated_x[i-1], cur_t)
      k2 = fn(des, simulated_x[i-1] + step_size*k1/2, cur_t + step_size/2)
      k3 = fn(des, simulated_x[i-1] + step_size*k2/2, cur_t + step_size/2)
      k4 = fn(des, simulated_x[i-1] + step_size*k3, cur_t + step_size)
      new_slope = (k1 + 2*k2 + 2*k3 + k4)/6

      if store:
         des_lin_accel.append(new_slope[3:6])
         des_rot_accel.append(new_slope[-3:])

      simulated_x[i] = simulated_x[i-1] + step_size*new_slope
      cur_t = cur_t + step_size
  return simulated_x

###############################################################################
# ---------------------------------MOTOR SPEEDS-------------------------------#
###############################################################################
#Creates motor speeds

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
   """
   creates motor speed points
   the first index of the speed indicates the time at which the speed should occur
   the second index is a 4 by 1 array of the four motor_speeds
   """
   time_and_speed = []
   time_and_speed.append((0, throttle_up*0))
   time_and_speed.append((0.1, throttle_up*6))
   time_and_speed.append((1, throttle_up*7))
   time_and_speed.append((1.05, pitch_forward*1.4))
   time_and_speed.append((1.1, throttle_up *3))
   time_and_speed.append((2.2, throttle_up*3))
   time_and_speed.append((2.4, pitch_backward*1.3))
   time_and_speed.append((3, throttle_up*3))
   time_and_speed.append((20, throttle_up*3))
   return time_and_speed

index = 0

def cts_motor_speed_fn(desired_motor_speeds, t):
   """
   makes the motor speeds continuous
   at a time_step in between two motor speed setpoints
   the motor speeds is calculated based on a weighted average of the 
   motor speeds between the two setpoints
   """
   global index
   if t >= desired_motor_speeds[index+1][0]:
      index +=1 
   prev_speed = desired_motor_speeds[index]
   next_speed = desired_motor_speeds[index+1]
   fn_output = prev_speed[1] + (t - prev_speed[0])/(next_speed[0] - prev_speed[0])*(next_speed[1] - prev_speed[1])
   return fn_output

##############################################################################
# ---------------------------------MAKE STATES-------------------------------#
##############################################################################
# makes num_iterations state vectors using the motor speeds from the previous section
# each state vector is a 12 by 1 vector representing p, v, angles, omega in that order

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

def state_func(des, x, t):
  """
  math to calculate derivative of state
  """
  p_dot = x[3:6]
  # makes the new acceleration and angular acceleration vectors
  cur_motors_abs = make_omega(cts_motor_speed_fn(des, t)) ##### CHANGE
  to_extract = np.vstack((-m*g*e3, -(np.cross(x[-3:].reshape(1,-1), (J@x[-3:]).reshape(1, -1))).reshape(-1, 1)))
  R = calculate_R(x[6], x[7], x[8])

  sub_step = np.block([[R, np.zeros((3,3))],[np.zeros((3,3)), np.eye(3)]])
  to_extract = to_extract + sub_step@F@cur_motors_abs
  v_dot = to_extract[:3]/m
  omega_dot = np.linalg.inv(J)@to_extract[-3:]
  angle_dot = make_M_inv(x[6], x[7], x[8])@x[-3:]
  output = np.vstack((p_dot, v_dot, angle_dot, omega_dot))
  return output

def make_M_inv(phi, theta, psi):
   """
   - makes the inverse of the M matrix, for calculating angle_dot
   - angle dot needs to be adjusted with this to convert from 
   body frame to global frame 
   """
   M = np.array([[1, 0, -math.sin(theta)],
                 [0, math.cos(phi), math.cos(theta)*math.sin(phi)],
                 [0, -math.sin(phi), math.cos(theta)*math.cos(phi)]])
   return np.linalg.inv(M)

des_lin_accel = [np.array([[0], [0], [0]])]
des_rot_accel = [np.array([[0], [0], [0]])]

##############################################################################
# ---------------------------------TEST STATES-------------------------------#
##############################################################################
#does not obey physical logic but that is ok!! 

def make_circles():
   state = []
   r = 2
   speed = 0.005
   height = 2
   for i in range(num_iterations+3):
      theta = i*speed
      velx = -r * math.sin(theta) * speed/step 
      vely = r* math.cos(theta) * speed/step 
      state.append(np.array([r*math.cos(theta), r*math.sin(theta), height, velx, vely,0,0,0,0,0,0,0]).reshape(12, 1))
   return state
# def make_circles():
#    state = []
#    r = 2
#    speed = 0.005
#    for i in range(num_iterations+3):
#       theta = i*speed
#       velx = -r * math.sin(theta) * speed/step 
#       vely = r* math.cos(theta) * speed/step 
#       state.append(np.array([r*math.cos(theta), r*math.sin(theta), 0, velx, vely,0,0,0,0,0,0,0]).reshape(12, 1))
#    return state

def constant():
   state = []
   for i in range(num_iterations+3):
      state.append(np.zeros((12,1)))
   return state 

##############################################################################
# ------------------------------ADAPTIVE CONTROLLER--------------------------#
##############################################################################
#seriously what in the everloving heck is going on?

def adaptive(desired_state, real_state):
  """
  The desired circular motion takes the form: x_dot = A* x + thet x where thet is a 
  4 by 4 array with all zeroes except in (2,2) and (3,3)
  We assume that the real motion takes the form x_dot = Ax + thet* x 
  A and theta are 4 by 4 arrays 
  """
  global kx
  global thet
  indices = [0, 1, 3, 4]
  des_state_rel = desired_state[indices] #relevant values of the real state 
  real_state_rel = real_state[indices] #relevant values of the desired state
  err = des_state_rel - real_state_rel
  assert err.shape == (4,1)

  kx_change = gamma*(err@real_state_rel.T)
  thet_change = - gamma*(err@real_state_rel.T)

  u = kx@real_state_rel + thet@real_state_rel
  assert u.shape == (4,1)

  kx = kx_change*step +  kx
  thet = thet_change*step + thet
  
  return u

################################################################################
# ---------------------------------PID CONTROLLER------------------------------#
################################################################################

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

   rwd = np.hstack((first_column, second_column, zwd))
   return rwd

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

def PID(desired_states, cur_real, t):
   """
   PID controller: calculates next force and torque
   Only for one iteration
   """
   iter = int(t/step)

   real_desired = desired_states[iter]

   update_desired = adaptive(desired_states[iter-1], cur_real) 

   cur_desired = np.vstack((real_desired[[0,1]] - update_desired[[0,1]], real_desired[2], real_desired[[3,4]] - update_desired[[2,3]], real_desired[5:]))
   assert cur_desired.shape == (12,1)

   if iter > num_iterations - 30:
         print(f'{iter = }')
         print(f'{real_desired = }')
         print(f'{update_desired = }')
         print(f'{cur_desired = }')

   real_w = cur_real[-3:]
   desired_w = cur_desired[-3:]
   phi, theta, psi = cur_real[6:9]

   # position and velocity errors
   ep =  cur_real[:3] - cur_desired[:3] 
   ev = cur_real[3:6] - cur_desired[3:6]

   # calculates desired applied force in the body frame 
   x_tilde = calculate_x_tilde(cur_desired[8]) 
   val = -kp*ep - kv*ev + m*g*e3 + m*des_lin_accel[iter]
   rwb = calculate_R(phi, theta, psi)
   fBz = np.dot(val.reshape(3,), (rwb@e3).reshape(3,))

   # calculates desired linear acceleration in the world frame (uses rwb)
   # also accounts for gravity
   final_lin_accel = (rwb@(fBz*e3)- m*g*e3)/m
   
   # calculate angle and rotational velocity errors  
   rwd = calculate_Rwd(val, x_tilde)
   temp_matrix = rwd.T@rwb -rwb.T@rwd
   er = 0.5*np.array([[temp_matrix[2][1]], [temp_matrix[0][2]], [temp_matrix[1][0]]])
   ew = real_w - rwb.T@rwd@desired_w

   # calculates desired torque 
   ax = cur_real[9].item() #creates matrix corresponding to cross product
   ay = cur_real[10].item()
   az = cur_real[11].item()
   cur_real_cross = np.array([[0, -az, ay], 
                              [az, 0, -ax], 
                              [-ay, ax, 0]])
   torqueB = -kr*er -kw*ew + np.cross(real_w.ravel(), (J@real_w).ravel()).reshape(3,1) - J@(cur_real_cross@rwb.T@rwd@desired_w - rwb.T@rwd@des_rot_accel[iter])
   assert torqueB.shape == (3,1) 

   # calculates desired angular acceleration 
   # also accounts for centrifugal force 
   ang_accel = np.linalg.inv(J)@(torqueB - (np.cross(real_w.ravel(), (J@real_w).ravel())).reshape(3,1)) #centrifugal force
   final_ang_accel = make_M_inv(phi, theta, psi)@(ang_accel) #converts to world frame

   final = np.vstack((cur_real[3:6], final_lin_accel, real_w, final_ang_accel))
   return final

###########################################################################
# --------------------------EVALUATION METRIC-----------------------------#
###########################################################################

def evaluate_performance(PID_states, desired_states):
   #average distance between real pos and desired pos
   total_sum = 0
   for i in range(num_iterations-2):
      total_sum = total_sum + np.linalg.norm(PID_states[i][:3] - desired_states[i][:3])
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
   plots position, velocity, angle, and omega graphs
   (3 per type, for the 3 axes)
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

def plot2D_both(toplot1, toplot2, save_as_pdf = False, name = "plot2D_both_AC.pdf"):
   """
   same as plot_mulaxis but plots for two different types of states 
   """
   fig, axs = plt.subplots(4, 3, layout = "constrained")
   labels = ["pos1", "pos2", "pos3", "vel1", "vel2", "vel3", "ang1", "ang2", "ang3", "ome1", "ome2", "ome3"]
   x_vals1 = []
   x_vals2 = []
   for i in range(0, len(toplot1)):
      x_vals1.append(i*step)
   for i in range(0, len(toplot2)):
      x_vals2.append(i*step)

   for i in range(12):
       y_vals1 = [vector[i] for vector in toplot1]
       y_vals2 = [vector[i] for vector in toplot2]
       axis_1 = math.floor(i/3)
       axis_2 = i%3
       axs[axis_1, axis_2].plot(x_vals1, y_vals1, label = "created")
       axs[axis_1, axis_2].plot(x_vals2, y_vals2, label = "desired")
       axs[axis_1, axis_2].set_title(labels[i])
   plt.legend()

   if save_as_pdf:
      fig.savefig(name) 

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

def plot3D_both(toplot1, toplot2, save_as_pdf = False, name = "plot3D_both_AC.pdf"):
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

   if save_as_pdf: 
      fig.savefig(name)

def plot_difference(real_states, desired_states, save_as_pdf = False, name = "plot_difference.pdf"):
   """
   plot the difference between desired state and real state
   """
   fig, axs = plt.subplots(2, 3, layout = "constrained")
   labels = ["pos1", "pos2", "pos3", "vel1", "vel2", "vel3"]
   x_vals = []
   difference_vals = []
   for i in range(0, min(len(desired_states), len(real_states))):
      x_vals.append(i*step)
      difference_vals.append(desired_states[i] - real_states[i])
   for i in range(6):
       axis_1 = math.floor(i/3)
       axis_2 = i%3
       y_vals = [vector[i] for vector in difference_vals]
       axs[axis_1, axis_2].plot(x_vals, y_vals, label = "created")
       axs[axis_1, axis_2].set_title(labels[i])
       
   if save_as_pdf: 
      fig.savefig(name)

def plot_distance(real_states, desired_states, save_as_pdf = False, name = "plot_distance.pdf"):
   fig = plt.figure()
   x_vals =[]
   for i in range(0, num_iterations-2):
      x_vals.append(i*step)

   y_vals = [] 
   y_zero = []
   for i in range(num_iterations-2):
      y_vals.append(np.linalg.norm(real_states[i][:3] - desired_states[i][:3]))
      y_zero.append(0)
   plt.plot(x_vals, y_vals)
   plt.plot(x_vals, y_zero)

   if save_as_pdf: 
      fig.savefig(name)

if __name__ == "__main__":
   #makes motor speeds
   desired_motor_speeds = make_motor_speeds_non_int()
   
   #makes desired state based on motor speeds 
   # desired_states = simulate_runge_kutta(state_func, des_initial, 0, step, num_iterations, desired_motor_speeds, True) 
   # ^^ makes states based on motor speed 
   # desired_states = constant()
   desired_states = make_circles()
   des_lin_accel = [np.array([[0], [0], [0]])] *num_iterations
   des_rot_accel = [np.array([[0], [0], [0]])] *num_iterations

   #PID states based on desired_states
   PID_states = simulate_runge_kutta(PID, real_initial, 0, step, num_iterations-2, desired_states)

   #  print(f'{PID_states = }')
   #  print(f'{desired_states = }')

   print(f'{evaluate_performance(PID_states, desired_states) = }')
   plot2D_both(PID_states, desired_states)
   plot3D_both(PID_states, desired_states)
   plot_difference(PID_states, desired_states)
   plot_distance(PID_states, desired_states)
   plt.show()

