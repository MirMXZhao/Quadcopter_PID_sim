"""
- Creates desired motor speeds
- Simulates quadcopter motion based on the desired motor speed 
  > creates an array of desired states
- The process above can be replaced with manual input of the states 
- The real quadcopter (with different initial states) follows the desired speeds using PID
  > creates an array of created states (converges to desired state
- simulates the crazyflie motion
"""
import numpy as np
from pycrazyswarm import *
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
num_iterations = 1000

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
real_initial = [1,0,0]

#global variables changed later on:
desired_motor_speeds = [0]

#Adaptive control parameters
gamma = 1.2
# A = np.zeros((4,4))
# A[0,2] = 1
# A[1,3] = 1
#initial conditions for kx, kr, thet 
kx = np.identity(6)*0.1
thet= [np.ones((6,1))*0 for _ in range(7)] 
thet2 = [np.ones((6,1))*0 for _ in range(2)]
random.seed(1) ## CHANGE THIS WHEN WANT TO TEST A DIFFERENT SET OF RANDOM NUMBERS
dist_rand = [random.uniform(0.1, 0.5) for _ in range (15)]
disturbance_const = np.array([0.5, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((-1,1))
#CHANGE TO CONSTANT DISTURBANCE TO SEE IT WORKING

stored_thet = [] 
stored_dist = [] 

#CRAZYFLIE INITIALIZATION
crazyflies_yaml = """
crazyflies:
- id: 1
  channel: 110
  initialPosition: [0, 0 , 0.0]
"""
swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml)
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs
cf = allcfs.crazyflies[0] #There should only be one crazyflie for this program
sleepRate = 15
num_cf = len(allcfs.crazyflies)

###############################################################################
# -----------------------------Find the best gamma----------------------------#
###############################################################################
def find_gamma(desired_state):
   """
   finds the best value of gamma by testing a bunch
   """
   lowest_eval = float("inf")
   lowest_gamma = 0
   stored_eval = []
   for i in range(21, 25):
      gamma = i*0.05
      real_states = cf_sim(gamma, desired_state)
      eval = evaluate_performance(real_states, desired_states)
      stored_eval.append((gamma, eval))
      print(f'{gamma = }')
      print(f'{eval = }')
      if eval < lowest_eval:
         lowest_gamma = gamma
         lowest_eval = eval
   print(real_states)
   return (lowest_gamma, stored_eval)

###############################################################################
# --------------------------RK4 + Crazyflie simulator-------------------------#
###############################################################################

def simulate_runge_kutta(fn, x0, t0, step_size, num_iters, des):
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

      des_lin_accel.append(new_slope[3:6])
      des_rot_accel.append(new_slope[-3:])

      simulated_x[i] = simulated_x[i-1] + step_size*new_slope
      cur_t = cur_t + step_size
  return simulated_x

def cf_sim(gamma, desired_state):
   """
   simulates crazyflie, incorporating adaptive control
   """
   recorded_states = []
   cur_desired = np.zeros((12,1))
   cf.cmdFullState(real_initial, [0,0,0], [0,0,0], 0, [0,0,0])
   timeHelper.sleep(0.3)
   for i in range(1,num_iterations):
      real_desired = desired_state[i]

      pos = cf.position()
      vel = (cur_desired[3:6].reshape(3,)).tolist()
      cur_state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]).reshape(-1,1)
      recorded_states.append(np.vstack((cur_state, desired_state[i-1][6:])))
      assert np.vstack((cur_state, desired_state[i-1][6:])).shape == (12,1)

      update_desired = adaptive(gamma, desired_state[i-1], cur_state)
      cur_desired = np.vstack((real_desired[[0,1,2,3,4,5]] + update_desired[[0,1,2,3,4,5]], real_desired[6:]))

      motion(np.add(cur_desired, disturbance(i)), [0,0,0])
   return recorded_states

def cf_sim_no_AC(desired_state):
   """
   crazyflie simulation without adaptive control 
   """
   recorded_states = []
   for i in range(num_iterations):
      cur_desired = desired_state[i]
      motion(np.add(cur_desired, disturbance(i)), [0,0,0])
      pos = cf.position()
      vel = (cur_desired[3:6].reshape(3,)).tolist()
      cur_state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]).reshape(-1,1)
      recorded_states.append(np.vstack((cur_state, cur_desired[6:])))
   return recorded_states

def motion(state, accel):
   """
   converts state to crazyflie motion
   """
   pos = (state[:3].reshape((3,))).tolist()
   vel = (state[3:6].reshape((3,))).tolist()
   acc = accel
   ome = (state[-3:].reshape((3,))).tolist()
   yaw = 0 
   # cf.cmdVelocityWorld(vel, yaw)
   cf.cmdFullState(pos, vel, acc, yaw, ome)
   # cf.cmdFullState([0,0,0], [0,0,0], [0,0,0], 0, [0,0,0])
   timeHelper.sleep(0.003)
   # timeHelper.sleepForRate(sleepRate)

def disturbance(iter):
   """
   continuous disturbance function
   """
   funcs = [math.cos, math.sin, math.cos, math.sin, math.cos]
   output = np.zeros((12,1))
   for i in range(5):
      output[0] += funcs[i](dist_rand[i]*iter*0.07) + 0.3
      output[1] += funcs[i](dist_rand[i+5]*iter*0.07) + 0.15
      output[2] += funcs[i](dist_rand[i+10]*iter*0.07) + 0.03
   assert output.shape == (12,1)
   stored_dist.append(output/2)
   return output/3

##############################################################################
# ---------------------------------TEST STATES-------------------------------#
##############################################################################
#does not obey physical logic but that is ok!! 

def make_circles():
   state = []
   r = 2
   speed = 0.01
   height = 2
   for i in range(num_iterations+3):
      theta = i*speed
      velx = -r * math.sin(theta) * speed/step 
      vely = r* math.cos(theta) * speed/step 
      state.append(np.array([r*math.cos(theta), r*math.sin(theta), height, velx, vely,0,0,0,0,0,0,0]).reshape(12, 1))
   return state

##############################################################################
# ------------------------------ADAPTIVE CONTROLLER--------------------------#
##############################################################################

def make_phi_x(real_state):
   arr = []
   for i in range(6):
      arr.append(np.identity(6)*real_state[i])
   return np.hstack((arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], np.identity(6)))

def adaptive(gamma, desired_state, real_state):
   """
   The desired circular motion takes the form: x_dot = A* x + thet x where thet is a 
   4 by 4 array with all zeroes except in (2,2) and (3,3)
   We assume that the real motion takes the form x_dot = Ax + thet* x 
   A and theta are 4 by 4 arrays 
   """
   global kx
   global thet
   indices = [0, 1, 2, 3, 4, 5]
   des_state_rel = desired_state[indices] #relevant values of the real state 
   real_state_rel = real_state[indices] #relevant values of the desired state
   err = des_state_rel - real_state_rel
   assert err.shape == (6,1)

   kx_change = gamma*(err@real_state_rel.T)
   kx = kx_change*step +  kx
   u = kx@real_state_rel
   # print(f'{kx = }')
   # print(f'{u = }')
   phis = real_state_rel.reshape(6,).tolist()
   phis.append(1)
   for i in range(7):
      thet_change = - gamma*phis[i]*(np.identity(6)@err)
      thet[i] = thet[i] + thet_change*step
      # print(f'{thet[i] =}')
      assert thet[i].shape == (6,1)
      u = u - phis[i]*np.identity(6)@thet[i]

   assert u.shape == (6,1)
   # print(f'{u = }')
   stored_thet.append(thet)
   return u

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
def plot2D(toplot, num_val):
   x_vals = []
   for i in range(0, len(toplot)):
      x_vals.append(i*step)

   for i in range(num_val):
      y_vals = [vector[i] for vector in toplot]
      plt.plot(x_vals, y_vals, label = i)

def plot2D_both(toplot1, toplot2, save_as_pdf = False, name = "plot2D_both_AC_cf.pdf"):
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

def plot3D_both(toplot1, toplot2, save_as_pdf = False, name = "plot3D_both_AC_cf.pdf"):
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

def plot_difference(real_states, desired_states, save_as_pdf = False, name = "plot_distance.pdf"):
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
   # Testing
   # desired_states = constant()
   desired_states = make_circles()

   # gamma_test = find_gamma(desired_states)
   # print(gamma_test)

   real_states = cf_sim(gamma, desired_states)
   # real_states = cf_sim_no_AC(desired_states)

   # print(f'{real_states = }')
   # print(f'{desired_states = }')

   #plotting + evaluation
   # plot2D(stored_thet, 5)
   plot2D(stored_dist, 3)
   plot2D_both(real_states, desired_states)
   plot3D_both(real_states, desired_states, False, "pl_AC_3D_disturb_vert.pdf")
   plot_distance(real_states, desired_states, False,  "pl_AC_dist_disturb_vert.pdf")
   plot_difference(real_states, desired_states)
   print(f'{evaluate_performance(real_states, desired_states) = }')
   plt.show()
