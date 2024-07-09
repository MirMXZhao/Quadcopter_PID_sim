import numpy as np
import ipdb 
from qpsolvers import solve_qp
import random
import math
import scipy
import matplotlib.pyplot as plt


######################################uwu####################################
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
num_iterations = 300

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
discretization = 0.05
desired_motor_speeds = []
# desired_states = []

real_initial = np.zeros((12,1))
# real_initial = np.random.uniform(low = -1, high = 2, size = (12,1))
real_initial[2] = 1
real_initial[0] = 1
real_initial[1] = 0
# real_initial[-9:] = 0
kp = 20
kv = 20
kr = 0.4
kw = 0.4

#######################################iwi####################################
# ---------------------------------MAKE STATES-------------------------------#
##############################################################################
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

def simulate_runge_kutta(fn, x0, t0, step_size, num_iters, store = False):
  """
  - RK4
  #used in the calculation of desired states 
  #as well as the calculation of the real states 
  """
  #Something about step_size is weird
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
      simulated_x[i] = simulated_x[i-1] + step_size/6*(k1 + 2*k2 + 2*k3 + k4)
      cur_t = cur_t + step_size
  return simulated_x

# desired_states = make_state() # this is the array of desired states
desired_states = [np.zeros((12,1))]*(num_iterations+2)
# print(f'{desired_states = }')

################################################################################
# ---------------------------------PID CONTROLLER------------------------------#
################################################################################
created_motor_speeds = []

des_lin_accel = [np.zeros((3,1))]*(num_iterations + 2)
des_rot_accel = [np.zeros((3,1))]*(num_iterations + 2)
# print(f'{des_lin_accel = }')
# print(f'{des_rot_accel = }')

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
   # print(f'{zwd = }')
   # print(f'{x_tilde=}')
   second_column = (np.cross(zwd.ravel(), x_tilde.ravel())/(np.linalg.norm(np.cross(zwd.ravel(), x_tilde.ravel())))).reshape(3,1)
   first_column = (np.cross(second_column.ravel(), zwd.ravel())/(np.linalg.norm(np.cross(second_column.ravel(), zwd.ravel())))).reshape(3,1)
   # print(f'{first_column = }')
   # print(f'{second_column = }')
   Rwd = np.hstack((first_column, second_column, zwd))
   # print(f'{Rwd = }')
   return Rwd

def PID(cur_real, t):
   """
   PID controller: calculates next force and torque
   Only for one iteration
   """
   #calculate the defined errors for position, velo, rot, and ang vel
   iter = int(t) #THIS IS NOT CORRECT (in general; it is correct in the constant case, however)
   print(f'{iter = }')
   cur_desired = desired_states[iter]
   ep =  - (cur_desired[:3] - cur_real[:3])
   ev = - (cur_desired[3:6] - cur_real[3:6])

   x_tilde = calculate_x_tilde(cur_desired[8]) #check if the last is actually yaw
   val = -kp*ep - kv*ev + m*g*e3 + m*des_lin_accel[iter+1] #CHECK if this is correct
   # print(f'{val = }')
   # print(f'{des_lin_accel[iter] = }')
   # print(f'{-kp*ep-kv*ev=}')
   # print(f'{m*des_lin_accel[iter]=}')
   # print(f'{val = }') #PRINT FOR DEBUGGING
   rwd = calculate_Rwd(val, x_tilde)
   rwb = calculate_R(cur_real[6], cur_real[7], cur_real[8])
   temp_matrix = rwd.T@rwb -rwb.T@rwd
   er = 0.5*np.array([[temp_matrix[2][1]], [temp_matrix[0][2]], [temp_matrix[1][0]]])

   ew = cur_real[-3:] - rwb.T@rwd@cur_desired[-3:]
   # print(f'{ep = }')
   # print(f'{ev = }')
   # print(f'{er = }')
#    print(f'{rwb = }')
#    print(f'{rwd = }')
#    print(f'{ew = }')
   fBz = val*rwb@e3
   ax = cur_real[9].item()
   ay = cur_real[10].item()
   az = cur_real[11].item()
   cur_real_cross = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
   torqueB = -kr*er -kw*ew + np.cross(cur_real[-3:].ravel(), (J@cur_real[-3:]).ravel()) - J@(cur_real_cross@rwb.T@rwd@cur_desired[-3:] - rwb.T@rwd@des_rot_accel[iter])
   # print(f'{np.linalg.inv(J)@torqueB=}')
   print(f'{torqueB =}')
   ang_accel = np.linalg.inv(J)@[torqueB - np.cross(cur_real[-3:].ravel(), (J@cur_real[-3:]).ravel())]
   print(f'{ang_accel =}')
   # print(f'{fBz/m = }')
   # print(f'{cur_real[3:6] = }')
   # print(f'{ang_accel[:, 0] = }')
   final = np.vstack((cur_real[3:6], (fBz - m*g*e3)/m, cur_real[-3:], ang_accel[:, 0].reshape(-1, 1)))
   # print(f'{final =}')
   return final

created_states = simulate_runge_kutta(PID, real_initial, 0, discretization, num_iterations-2)
# print(f'{created_states = }')

#############################################################################
# ---------------------------------CONVERSION-------------------------------#
#############################################################################
#input force and torque, convert to motor speeds
def convert_state_to_speed(f_and_t):
   """
   f and t is a list of 6 by 1 column vectors
   the first 3 rows represents the force (the first 2 rows will be 0)
   the last 3 rows represent the torque
   """
   F_inv = np.linalg.inv(F[-4:, :])
   omega_control = []
   for ele in f_and_t:
       omega_control.append(F_inv@ele)
   return omega_control

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
   ax.plot3D(x_val1, y_val1, z_val1, 'orange', label = "real")
   ax.plot3D(x_val2, y_val2, z_val2, 'purple', label = "desired")

if __name__ == "__main__":
   with ipdb.launch_ipdb_on_exception(): 
   #stuff
   # print(f'{evaluate_performance() = }')
  
      #testing:
      # plot_1axis(state)
      # plot_mulaxis(state)
      # plot3D(state)

      #testing PID
      # plot_mulaxis(created_states, num_iterations-1)
      # plot_mulaxis(desired_states)
      print(f'{evaluate_performance() = }')
      print(created_states)
      print(desired_states)
      plot2D_both(created_states, desired_states)
      # plot3D(created_states)
      plot3D_both(created_states, desired_states)
      plt.show()

