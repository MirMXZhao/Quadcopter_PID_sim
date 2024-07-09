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

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs

num_cf = len(allcfs.crazyflies)
actual_data = np.zeros((num_cf, num_cf, 100))
pos_data = np.zeros((num_cf, 100))

#uneven circular motion for goals
def goalMotion(totalTime, radius, kPosition, time_i):
    startTime = timeHelper.time()
    pos = [None]*num_cf
    startPos = [None]*num_cf
    center_circle = [None]*num_cf
    goal = np.zeros((num_cf, 3))

    for i in range(num_cf):
        pos[i] = allcfs.crazyflies[i].position()
        startPos[i] = allcfs.crazyflies[i].initialPosition + np.array([0, 0, Z])
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    # time = timeHelper.time() - startTime
    time = time_i*0.1
    for i in range(num_cf):
        omega = 0.5* np.pi / totalTime[i]
        goal[i, :] = center_circle[i] + radius[i] * np.array([np.cos(omega * time), np.sin(omega * time), 0])
        timeHelper.sleepForRate(sleepRate)

    return goal #goal encodes desired position

# problem: minimize v^2 given 
#1. (v - v_goal)^(T)(p - p_goal) <= -1/2(p - p_goal)^(T)(p-p_goal) +1/2*0.01^2
#2. (vi-vj)^(T)(pi-pj) <= 1/2(|pi-pj|^2 - d^2)
def dontCrash(vprev, goal, time_i):
    #position matrix: used in further 
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
        G[x][3*i] = pos[i][0] - goal[i][0]
        G[x][3*i+1] = pos[i][1] - goal[i][1]
        G[x][3*i+2] = pos[i][2] - goal[i][2]
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
        vi_cur = [vprev[3*i], vprev[3*i+1], vprev[3*i+2]]
        h[x] = -math.dist(pos[i], goal[i])/2 +0.01*0.01/2 + np.dot(pos[i] - goal[i], vi_cur)
        x+=1
    for i in range(num_cf):
        for j in range(i+1, num_cf): #bound for condition 2
            h[x] = (sradius*sradius - math.dist(pos[i], pos[j])*math.dist(pos[i], pos[j]))/2
            actual_data[i][j][time_i] = math.dist(pos[i], pos[j]) #takes position data for future plotting
            x+=1

    for i in range(num_cf): #takes position data for future plotting
        pos_data[i][time_i] = math.dist(pos[i], goal[i])
    
    v = solve_qp(P,q,G,h,A=None,b=None, solver = "daqp") 

    # print("goal ", + goal)
    # print("pos", + pos)
    # print("G", + G)
    # print("h", + h)
    # print("v", + v)

    return v

def plot():
    #graphs pairwise distance between drones
    fig, axs = plt.subplots(num_cf-1, num_cf-1)
    plt.figure()
    for x in range(num_cf):
        for y in range(x+1, num_cf):
            axs[x,y-1].plot(range(0,100), actual_data[x][y]) 
            axs[x,y-1].plot(range(0,100), [sradius]*100)
            axs[x,y-1].set_title(str(x+1)+ " and " + str(y+1)) 

    #graphs distance between drone and goal position 
    fig2, axs2 = plt.subplots(num_cf)
    plt.figure()
    for x in range(num_cf):
        axs2[x].plot(range(0,100), pos_data[x])  
        axs2[x].set_title(str(x+1))
    
    plt.show()

def motion(v):
    for i in range(len(allcfs.crazyflies)):
        allcfs.crazyflies[i].cmdVelocityWorld([v[3*i],v[3*i+1],v[3*i+2]], yawRate=0) # change this so it fits with the definition of v
    timeHelper.sleep(0.02)

if __name__ == "__main__":
    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)
    timeHelper.sleep(15)

    i = 0
    vprev = np.zeros(num_cf*3)
    while i < 100:
        goal = goalMotion(totalTime=[2.146,-0.73, 1, -1.67], radius=[4,2,3,3.4], kPosition=1, time_i=i)
        v = dontCrash(vprev, goal, time_i = i)
        motion(v)
        vprev = v
        i+=1 
    plot()
