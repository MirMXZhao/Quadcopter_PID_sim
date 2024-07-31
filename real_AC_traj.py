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
import os
import datetime
import pickle

import matplotlib.pyplot as plt

#############################################################################
# ---------------------------------CONSTANTS--------------------------------#
#############################################################################
#NUMBER OF ITERATIONS 
num_iterations = 9000

#DISCRETIZATION
step = 0.003

#desired starting state
des_initial = np.zeros((12, 1))

#starting state of the real drone
real_initial = np.zeros((12,1))
real_initial[0] = 0
real_initial[1] = 0
real_initial[2] = 0

#global variables changed later on:
desired_motor_speeds = [0]

#Adaptive control parameters
gamma = 0.6
# A = np.zeros((4,4))
# A[0,2] = 1
# A[1,3] = 1
#initial conditions for kx, kr, thet 
kx = np.identity(6)*0.01
thet= np.ones((12,1))*0.0
random.seed(1) ## CHANGE THIS WHEN WANT TO TEST A DIFFERENT SET OF RANDOM NUMBERS
dist_rand = [random.uniform(0.1, 0.5) for _ in range (15)]
disturbance_const = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((-1,1))
#CHANGE TO CONSTANT DISTURBANCE TO SEE IT WORKING BEST

#Trajectory parameter
alpha = 0.8

stored_thet = [] 
stored_dist = []
stored_gamma = []

#CRAZYFLIE INITIALIZATION
swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs
cf = allcfs.crazyflies[0] #There should only be one crazyflie for this program
sleepRate = 15
num_cf = len(allcfs.crazyflies)

#DESIRED MOTION
r = 0.7
speed = 0.003
height = 0.6


###############################################################################
# -----------------------------Crazyflie simulator----------------------------#
###############################################################################

def cf_sim(desired_state):
   """
   simulates crazyflie, incorporating adaptive control
   """
   global last_traj_pos
   recorded_states = []
   cur_desired = np.zeros((12,1))
   traj_prev = False
   for i in range(1,num_iterations):
      real_desired = desired_state[i]

      pos = cf.position()
      vel = (cur_desired[3:6].reshape(3,)).tolist()
      cur_state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]).reshape(-1,1)
      recorded_states.append(np.vstack((cur_state, desired_state[i-1][6:])))
      assert np.vstack((cur_state, desired_state[i-1][6:])).shape == (12,1)

      #Trajectory Update
      if np.linalg.norm(cur_state[:3]- desired_state[i-1][:3]) > 0:
         if not traj_prev:
            last_traj_pos = cur_state[:3]
         traj_prev = True
         new_pos = updateTraj(desired_state[i][:6])
         new_desired = np.vstack((new_pos, desired_state[i][3:]))
         # print(new_pos)
      else: 
         new_desired = real_desired

      #Adaptive Control 
      update_desired = adaptive(new_desired, cur_state)
      cur_desired = np.vstack((new_desired[[0,1,2,3,4,5]] + update_desired[[0,1,2,3,4,5]], new_desired[6:]))

      motion(np.add(cur_desired, disturbance_const), [0,0,0])

   return recorded_states

def cf_sim_no_AC(desired_state):
   """
   crazyflie simulation without adaptive control 
   """
   recorded_states = []
   for i in range(num_iterations):
      cur_desired = desired_state[i]
      pos = cf.position()
      motion(np.add(cur_desired, sim_wind(pos)), [0,0,0])
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
      output[0] += funcs[i](dist_rand[i]*iter*0.02) + 0.2
      output[1] += funcs[i](dist_rand[i+5]*iter*0.02) + 0.1
      output[2] += funcs[i](dist_rand[i+10]*iter*0.02) + 0.02
   assert output.shape == (12,1)
   stored_dist.append(output/2)
   return output/3

std_dev = r/2
mean = 0 

def sim_wind(x):
   x_val = x[0]
   print(x_val)
   normal_val = 1/(std_dev*math.sqrt(2*math.pi))*(math.e**(-(x_val - mean)**2/(2*std_dev**2)))
   print(normal_val)
   return normal_val
##############################################################################
# ---------------------------------TEST STATES-------------------------------#
##############################################################################
#does not obey physical logic but that is ok!! 

def make_circles():
   state = []
   for i in range(num_iterations+3):
      theta = i*speed
      velx = -r * math.sin(theta) * speed/step 
      vely = r* math.cos(theta) * speed/step 
      state.append(np.array([r*math.cos(theta), r*math.sin(theta), height, velx, vely,0,0,0,0,0,0,0]).reshape(12, 1))
   return state

def constant():
   state = []
   for i in range(num_iterations+3):
      state.append(np.zeros((12,1)))
   return state 
##############################################################################
# ------------------------------ADAPTIVE CONTROLLER--------------------------#
##############################################################################

def make_phi_x(real_state):
   arr = []
   for i in range(3):
      arr.append(np.identity(3)*real_state[i])
   return np.hstack((arr[0], arr[1], arr[2], np.identity(3)))

def adaptive(desired_state, real_state):
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

  phi_x = make_phi_x(real_state_rel)
  assert phi_x.shape == (3, 12)

  kx_change = gamma*(err@real_state_rel.T)
  thet_change = - gamma*(phi_x.T@err[:3])
  assert thet_change.shape == (12,1)
  
  kx = kx_change*step +  kx
  thet = thet_change*step + thet

  phithet = np.vstack((phi_x@thet, np.array([[0], [0], [0]])))
  u = kx@real_state_rel - phithet
  assert u.shape == (6,1)

  stored_thet.append(thet)
  return u

last_traj_pos = np.array([[0], [0], [0]])

def updateTraj(desired_state):
   global last_traj_pos
   traj_change = -alpha*last_traj_pos + alpha*desired_state[:3] + desired_state[-3:]
   assert traj_change.shape == (3,1)
   new_traj = traj_change*step + last_traj_pos
   last_traj_pos = new_traj
   return new_traj

###########################################################################
# --------------------------EVALUATION METRIC-----------------------------#
###########################################################################

def evaluate_performance(real_states, desired_states):
   #average distance between real pos and desired pos
   total_sum = 0
   for i in range(num_iterations-2):
      total_sum = total_sum + np.linalg.norm(real_states[i][:3] - desired_states[i][:3])
   return total_sum/(num_iterations-2)

###########################################################################
# ------------------------------PLOTTING + DATA---------------------------#
###########################################################################
def plot2D(toplot, num_val):
   x_vals = []
   for i in range(0, len(toplot)):
      x_vals.append(i*step)

   fig, ax = plt.subplots()
   for i in range(num_val):
      y_vals = [vector[i] for vector in toplot]
      ax.plot(x_vals, y_vals, label = i)
   
   ax.set_title("THETA VALUES")
   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Value")

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
   ax.set_aspect("equal")

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
   fig, ax = plt.subplots()

   x_vals =[]
   for i in range(0, num_iterations-2):
      x_vals.append(i*step)
   y_vals = [] 
   y_zero = []
   for i in range(num_iterations-2):
      y_vals.append(np.linalg.norm(real_states[i][:3] - desired_states[i][:3]))
      y_zero.append(0)
   ax.plot(x_vals, y_vals)
   ax.plot(x_vals, y_zero)

   ax.set_title("Error")
   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Distance (m)")

   if save_as_pdf: 
      fig.savefig(name)

def create_folder(base_path = "./data", prefix = "run_"):
   """
   creates a new folder based on the timestamp of the run
   """
   timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
   folder_name = f"{prefix}{timestamp}"
   folder_path = os.path.join(base_path, folder_name)
   os.makedirs(folder_path, exist_ok = True)
   return folder_path

def save_pickle(folder_path, filename, data):
   pickle_path = os.path.join(folder_path, filename)
   with open(pickle_path, 'wb') as f:
      pickle.dump(data, f)
   return pickle_path
   
if __name__ == "__main__":
    plot = True # EDIT THIS TO CONTROL WHETHER IT PLOTS
    try:
        AC = False
        # Testing
        # desired_states = constant()
        desired_states = make_circles()

        if AC: 
           real_states = cf_sim(desired_states)
        else:
           allcfs.takeoff(targetHeight=height, duration=1.0+height)
           timeHelper.sleep(1.5+height)
           cf.goTo([r,0,height], 0, 2)
           timeHelper.sleep(2)
           real_states = cf_sim_no_AC(desired_states)

        #  print(f'{PID_states = }')
        #  print(f'{desired_states = }')
        for i in range(num_cf):
            allcfs.crazyflies[i].notifySetpointsStop() #changes to high level commands
        allcfs.land(targetHeight=0.04, duration=2.5)
        timeHelper.sleep(3+1)
   
    except KeyboardInterrupt: #forces land if cancelled
        # try to use timeHelper.isShutdown()
        print("oops")
        for i in range(num_cf):
            allcfs.crazyflies[i].notifySetpointsStop() #changes to high level commands

        allcfs.land(targetHeight=0.04, duration=2.5)
        timeHelper.sleep(3+1)

    #plotting + evaluation
    plot2D(stored_thet, 5)
    # plot2D(stored_dist, 3)
    plot2D_both(real_states, desired_states)
    print(f'{evaluate_performance(real_states, desired_states) = }')

    #creates a new folder and saves all data
    folder_path = create_folder()
    pl3D_path = os.path.join(folder_path, "3D.pdf")
    pldistance = os.path.join(folder_path, "distance.pdf")
    pldifference = os.path.join(folder_path, "difference.pdf")
    pl2D = os.path.join(folder_path, "2D.pdf")
    plot3D_both(real_states, desired_states, plot, pl3D_path)
    plot_distance(real_states, desired_states, plot,  pldistance)
    plot_difference(real_states, desired_states, plot, pldifference)
    plot2D_both(real_states, desired_states, plot, pl2D)
    all_data = [real_states, desired_states, stored_thet]
    save_pickle(folder_path, "pickle.pkl", all_data)
    plt.show()

