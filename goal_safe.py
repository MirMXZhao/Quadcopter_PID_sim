#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
from qpsolvers import solve_qp
import random 
import math
import matplotlib.pyplot as plt
# from IPython import get_ipython

Z = 1.0
sleepRate = 30
sradius = 1
# p_goal = ([-1.5, -1.5, 1.], [1.5, 1.5, 1.])
p_goal = ([-1.5, -1.5, 1.], [-1.5, 1.5, 1.], [1.5, -1.5, 1.], [1.5, 1.5, 1.])
p_goal = np.array([p_goal]).reshape(4, 3)
p_goal[:, :2] = p_goal[:, :2] + 0.1* np.random.randn(4, 2) # this adds randomness to the goal location: prevents them from getting stuck in deadlock position. 

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs

num_cf = len(allcfs.crazyflies)
data = np.zeros((num_cf, num_cf, 100))

# problem: minimize v^2 given 
#1. (v)^(T)(p - p_goal) <= -1/2(p - p_goal)^(T)(p-p_goal) +1/2*0.01^2
#2. (vi-vj)^(T)(pi-pj) <= 1/2(|pi-pj|^2 - d^2)
def dontCrash():
    #position matrix: used in further calculations
    pos = np.zeros((num_cf, 3))
    i = 0
    for cf in allcfs.crazyflies:
        pos[i, :] = cf.position()
        i = i+1

    P = np.identity(3*num_cf)

    q = np.zeros((num_cf)*3) # no q because there is no command velocity      

    dim = int(num_cf*(num_cf-1)/2) 
    G = np.zeros((dim + num_cf, num_cf*3))
    x = 0
    for i in range(num_cf): # This describes condition 1 (getting to the goal position)
        G[x][3*i] = pos[i][0] - p_goal[i][0]
        G[x][3*i+1] = pos[i][1] - p_goal[i][1]
        G[x][3*i+2] = pos[i][2] - p_goal[i][2]
        x+=1
    for i in range(num_cf): # This describes condition 2 (not crashing)
        for j in range(i+1, num_cf):
            dist_ij = np.linalg.norm(pos[i] - pos[j])
            if dist_ij > 2 * sradius:
                x = x+1
                continue
            G[x][3*j] = pos[i][0] - pos[j][0]
            G[x][3*j+1] = pos[i][1] - pos[j][1]
            G[x][3*j+2] = pos[i][2] - pos[j][2]
            G[x][3*i] = -pos[i][0] + pos[j][0]
            G[x][3*i+1] = -pos[i][1] + pos[j][1]
            G[x][3*i+2] = -pos[i][2] + pos[j][2]
            x+=1

    h = np.zeros(dim + num_cf)
    x=0
    for  i in range(num_cf): #bound for condition 1
        h[x] = -math.dist(pos[i], p_goal[i])/2 +0.05*0.05/2
        x+=1
    for i in range(num_cf):
        for j in range(i+1, num_cf): #bound for condition 2
            h[x] = (sradius*sradius - math.dist(pos[i], pos[j])*math.dist(pos[i], pos[j]))/2
            x+=1
    # print("pos", + pos)
    # print("G", + G)
    # print("h", + h)
    
    v = solve_qp(P,q,G,h,A=None,b=None, solver = "daqp")
    # if v is None: #adds randomness in deadlock scenario. interchangeable with code in line 15
    #     v = np.random.uniform(low = -0.5, high = 0.5, size = (num_cf*3,))*0.01 
    #     # print("CHANGE", + v) 

    # v = np.clip(v, -1, 1) #prevents it from going super far apart due to the deadlock
    
    # print("v", + v)
    # print()
    # motion(v, allcfs)

    return v

def plot(data, num_cf):
    fig, axs = plt.subplots(num_cf-1, num_cf-1)
    plt.figure()
    for x in range(num_cf):
        for y in range(x+1, num_cf):
            axs[x,y-1].plot(range(0,100), data[x][y])
            axs[x,y-1].set_title(str(x+1)+ " and " + str(y+1)) 
    plt.show()

def motion(v):
    for i in range(len(allcfs.crazyflies)):
        allcfs.crazyflies[i].cmdVelocityWorld([v[3*i],v[3*i+1],v[3*i+2]], yawRate=0) # change this so it fits with the definition of v
    timeHelper.sleep(0.1)

if __name__ == "__main__":
    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    i = 0
    while i < 200:
        v = dontCrash()
        motion(v)
        i+=1 
    # plot(data, num_cf)
