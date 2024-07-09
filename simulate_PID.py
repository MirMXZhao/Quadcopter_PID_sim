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
num_iterations = 20

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
discretization = 0.01
desired_motor_speeds = []
# desired_states = []

real_initial = np.zeros((12,1))
# real_initial = np.random.uniform(low = -1, high = 2, size = (12,1))
real_initial[2] = 200
real_initial[0] = 0
real_initial[1] = 0
# real_initial[-9:] = 0
kp = 0.8
kv = 0.3
kr = 0.2
kw = 0.3


#######################################owo#####################################
# ---------------------------------MOTOR SPEEDS-------------------------------#
###############################################################################


def make_motor_speeds():
   """
   - returns a list of 2*num_iterations motor speeds
   - motor speeds are a column vector of size 4
   """
   stored = []
   throttle_control(stored, 5.5, 5)
   roll_control(stored, True, 2, 2)
   # roll_control(stored, False, 1.5, 3)
   throttle_control(stored, 5.5, 60)
   # # throttle_control((m*g/(4*cf))**0.5, 5)
   # roll_control(stored, False, 2, 3)
   # throttle_control(stored, 3, 16)
   return stored

def throttle_control(stored, mag, reps):
   """
   - up: A boolean. if True, want to go up. Otherwise down.
   - mag: a float representing the strength of the control
   """
   for i in range(reps):
       new_thrust = np.array([[1],[1],[1],[1]])*mag
       stored.append(new_thrust)

def roll_control(stored, left, mag, reps):
   """
   - left: A boolean. if True, rolls to the left. Otherwise right
   - mag: a float representing the strength of the control
   - reps: number of times to perform this action
   """
   for i in range(min(reps, 10)):
       if left:
           new_roll = np.array([[1.1], [1], [1], [1.1]])*mag
       else:
           new_roll = np.array([[1], [1.1], [1.1], [1]])*mag
       stored.append(new_roll)

def pitch_control(stored, rotate_CW, mag, reps):
   """
   - forward: A boolean. if True, ptiches forwards. Otherwise backwards
   - mag: a float representing the strength of the control
   - reps: number of times to perform this action
   """
   for i in range(min(reps, 10)):
       if rotate_CW:
           new_pitch = np.array([[1.5], [1], [1.5], [1]])*mag
       else:
           new_pitch = np.array([[1], [1.5], [1], [1.5]])*mag
       stored.append(new_pitch)


def yaw_control(stored, forward, mag, reps):
   """
   - forward: A boolean. if True, ptiches forwards. Otherwise backwards
   - mag: a float representing the strength of the control
   - reps: number of times to perform this action
   """
   for i in range(min(reps, 10)):
       if forward:
           new_yaw = np.array([[1], [1], [1.5], [1.5]])*mag
       else:
           new_yaw = np.array([[1.5], [1.5], [1], [1]])*mag
       stored.append(new_yaw)

desired_motor_speeds = make_motor_speeds()

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
   store_state = simulate_runge_kutta(state_func, initial, 0, 1, num_iterations)
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
  # makes the new acceleration and angular acceleration vectors
  cur_motors_abs = make_omega(desired_motor_speeds[int(t)])
  to_extract = np.vstack((-m*g*e3, -(np.cross(x[-3:].reshape(1,-1), (J@x[-3:]).reshape(1, -1))).reshape(-1, 1)))
  R = calculate_R(x[6], x[7], x[8])
#    print(f'{F@cur_motors_abs = }')
  sub_step = np.block([[R, np.zeros((3,3))],[np.zeros((3,3)), np.eye(3)]])
  to_extract = to_extract + sub_step@F@cur_motors_abs
#    print(f'{to_extract = }')
  v_dot = to_extract[:3]/m
  omega_dot = np.linalg.inv(J)@to_extract[-3:]
  angle_dot = make_M_inv(x[6], x[7], x[8])@x[-3:]
  output = np.vstack((p_dot, v_dot, np.mod(angle_dot, 30*math.pi), np.mod(omega_dot, 30*math.pi)))
  return output

def make_M_inv(phi, theta, omega):
   """
   - makes the inverse of the M matrix, for calculating angle_dot
   """
   M = np.array([[1, 0, -math.sin(theta)],
                 [0, math.cos(phi), math.cos(theta)*math.sin(phi)],
                 [0, -math.sin(phi), math.cos(theta)*math.cos(phi)]])
   return np.linalg.inv(M)


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

desired_states = make_state() # this is the array of desired states
print(f'{desired_states = }')

################################################################################
# ---------------------------------PID CONTROLLER------------------------------#
################################################################################
created_motor_speeds = []

def calculate_accel(): # SOMETHING IS WRONG
   """
   Calculates positional acceleration (of the desired motion)
   This uses helper functions from the previous section (#MAKE STATES#)
   """
   simulated_lin_accel = [None]*num_iterations
   simulated_lin_accel[0] = np.array([[0], [0], [0]])
   simulated_rot_accel = [None]*num_iterations
   simulated_rot_accel[0] = np.array([[0], [0], [0]])
   for i in range(1, num_iterations):
      new_state = state_func(desired_states[i-1], i)
      print(i)
      simulated_lin_accel[i] = new_state[3:6]
      simulated_rot_accel[i] = new_state[-3:]
   return [simulated_lin_accel, simulated_rot_accel]

# def calculate_accel():
#    simulated_lin_accel = [None]*num_iterations
#    simulated_lin_accel[0] = np.array([[0], [0], [0]])
#    simulated_rot_accel = [None]*num_iterations
#    simulated_rot_accel[0] = np.array([[0], [0], [0]])
#    for i in range(1, num_iterations):
#       new_state = desired_states[i] - desired_states[i-1]
#       simulated_lin_accel[i] = new_state[3:6] - m*g*e3 
#       simulated_rot_accel[i] = new_state[-3:]
#    return [simulated_lin_accel, simulated_rot_accel]

des_accel = calculate_accel()
des_lin_accel = des_accel[0]
print(f'{des_lin_accel = }')
des_rot_accel = des_accel[1]
# print(f'{des_lin_accel = }')
# print(f'{des_rot_accel = }')

# def calculate_acceleration(): 
# THIS DOESNT WORK BECAUSE OF THE array formatting and im not sure why
# THIS MIGHT BE INCORRECT ??? depends on how state_func should be used (should be correct)
#    """
#    Calculates positional acceleration (of the desired motion)
#    This uses helper functions from the previous section (#MAKE STATES#)
#    """
#    simulated_accel = np.zeros((2, num_iterations))
#    for i in range(1, num_iterations):
#       new_state = state_func(desired_states[i-1], 2*(i-1))
#       simulated_accel[0][i] = new_state[3:6]
#       simulated_accel[1][i] = new_state[-3:]
#    return simulated_accel

# desired_accel = calculate_acceleration()
# des_lin_accel = desired_accel[0]
# des_ang_accel = desired_accel[1]
# print(f'{desired_acceleration=}')

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
   iter = int(t) #WRONG WRONG WRONG !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
   print(f'{ew = }')
   fBz = val*rwb@e3
   ax = cur_real[9].item()
   ay = cur_real[10].item()
   az = cur_real[11].item()
   cur_real_cross = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
   torqueB = -kr*er -kw*ew + np.cross(cur_real[-3:].ravel(), (J@cur_real[-3:]).ravel()) - J@(cur_real_cross@rwb.T@rwd@cur_desired[-3:] - rwb.T@rwd@des_rot_accel[iter])
   # print(f'{np.linalg.inv(J)@torqueB=}')
   print(f'{torqueB =}')
   ang_accel = np.linalg.inv(J)@(torqueB- np.cross(cur_real[-3:].ravel(), (J@cur_real[-3:]).ravel()))
   print(f'{ang_accel =}')
   # print(f'{fBz/m = }')
   # print(f'{cur_real[3:6] = }')
   # print(f'{ang_accel[:, 0] = }')
   final = np.vstack((cur_real[3:6], (fBz - m*g*e3)/m, cur_real[-3:], ang_accel[:, 1].reshape(-1, 1)))
   # print(f'{final =}')
   return final

created_states = simulate_runge_kutta(PID, real_initial, 0, 1, num_iterations-2)
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
   ax.plot3D(x_val2, y_val2, z_val2, 'blue', label = "desired")

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
      print(f'{created_states = }')
      print(f'{desired_states = }')
      print(f'{evaluate_performance() = }')
      plot2D_both(created_states, desired_states)
      # plot3D(created_states)
      plot3D_both(created_states, desired_states)
      plt.show()

