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

# walls = [3.3, -1.6, 4, -2.3]
walls = [6, -10, 10, -10]
wallsradius = 0.5
#uneven circular motion for goals
def goalMotion(totalTime, radius,  time_i):
    startTime = timeHelper.time()
    pos = [None]*num_cf
    startPos = [None]*num_cf
    center_circle = [None]*num_cf
    goal = np.zeros((2*num_cf, 3))

    for i in range(num_cf):
        pos[i] = allcfs.crazyflies[i].position()
        startPos[i] = allcfs.crazyflies[i].initialPosition + np.array([0, 0, Z])
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    # time = timeHelper.time() - startTime
    speed = 0.1 #makes it go slower
    time = time_i*speed
    for i in range(num_cf):
        omega = 0.5* np.pi / totalTime[i]
        vx = -radius[i] * omega * np.sin(omega * (time + speed))  
        vy = radius[i] * omega * np.cos(omega * (time + speed)) #CHANGED from time to time - speed >> worked????
        goal[2*i, :] = np.array([vx, vy, 0])
        goal[2*i+1, :] = center_circle[i] + radius[i] * np.array([np.cos(omega * time), np.sin(omega * time), 0])
        timeHelper.sleepForRate(sleepRate)

    return goal #goal encodes desired position in odd indices and desired velocity in even indices

# problem: minimize v^2 given 
#1. (v - v_goal)^(T)(p - p_goal) <= -1/2(p - p_goal)^(T)(p-p_goal) +1/2*0.01^2
#2. (vi-vj)^(T)(pi-pj) <= 1/2(|pi-pj|^2 - d^2)
#3. (v_x)(p_x - w_x) >= 1/2(|p_x - w_x|^2 - d^2)
def dontCrash(goal, time_i):
    #position matrix: used in further 
    pos = np.zeros((num_cf, 3))
    i = 0
    for cf in allcfs.crazyflies:
        pos[i, :] = cf.position()
        i = i+1

    P = np.identity(3*num_cf)

    q = np.zeros((num_cf)*3) # no q because there is no command velocity      

    dim = int(num_cf*(num_cf-1)/2) 
    G = np.zeros((num_cf + num_cf*4, num_cf*3))
    x = 0
    for i in range(num_cf): # This describes condition 1 (getting to the goal position)
        G[x][3*i] = pos[i][0] - goal[2*i+1][0]
        G[x][3*i+1] = pos[i][1] - goal[2*i+1][1]
        G[x][3*i+2] = pos[i][2] - goal[2*i+1][2]
        x+=1
    # for i in range(num_cf): # This describes condition 2 (not crashing)
    #     for j in range(i+1, num_cf):
    #         dist_ij = np.linalg.norm(pos[i] - pos[j])
    #         if dist_ij > 2 * sradius:
    #             x = x+1
    #             continue
    #         G[x][3*j] = pos[i][0] - pos[j][0]
    #         G[x][3*j+1] = pos[i][1] - pos[j][1]
    #         G[x][3*j+2] = pos[i][2] - pos[j][2]
    #         G[x][3*i] = -pos[i][0] + pos[j][0]
    #         G[x][3*i+1] = -pos[i][1] + pos[j][1]
    #         G[x][3*i+2] = -pos[i][2] + pos[j][2]
    #         x+=1
    for i in range(num_cf): # describes condition 3 (avoiding walls)
        for j in range(4):
            if j < 2:
                G[x][3*i+1] = walls[j]- pos[i][1] 
            else:
                G[x][3*i] =  walls[j]- pos[i][0]
            x+=1 

    
    h = np.zeros(num_cf + num_cf*4)
    x=0
    for  i in range(num_cf): #bound for condition 1
        h[x] = - math.dist(pos[i], goal[2*i+1])/2 + np.dot((pos[i] - goal[2*i+1]), goal[2*i])
        x+=1
    # for i in range(num_cf):
    #     for j in range(i+1, num_cf): #bound for condition 2,
    #         h[x] = (sradius*sradius - math.dist(pos[i], pos[j])*math.dist(pos[i], pos[j]))/2
    #         actual_data[i][j][time_i] = math.dist(pos[i], pos[j]) #collects data for plotting later
    #         x+=1,
    for i in range(num_cf):
        for j in range(4):
            if j<2:
                h[x] = - (wallsradius*wallsradius - (pos[i][1]-walls[j])*(pos[i][1]-walls[j]))/2 
            else:
                h[x] = - (wallsradius*wallsradius - (pos[i][0]-walls[j])*(pos[i][0]-walls[j]))/2
            x+=1

    print(time_i)
    print("goal ", + goal)
    print("pos", + pos)
    print("G", + G)
    print("h", + h)
    for i in range(num_cf): #takes position data for future plotting
        pos_data[i][time_i] = math.dist(pos[i], goal[2*i+1])
        print(pos_data[i][time_i])
    
    v = solve_qp(P,q,G,h,A=None,b=None, solver = "daqp") 
    v = np.clip(v, -2, 2)
    
    
    print("v", + v)

    return v

def plot():
    #graphs pairwise distance between drones
    # fig, axs = plt.subplots(num_cf-1, num_cf-1)
    # plt.figure()
    # for x in range(num_cf):
    #     for y in range(x+1, num_cf):
    #         axs[x,y-1].plot(range(0,100), actual_data[x][y]) 
    #         axs[x,y-1].plot(range(0,100), [sradius]*100)
    #         axs[x,y-1].set_title(str(x+1)+ " and " + str(y+1)) 

    #graphs distance between drone and goal position 
    fig2, axs2 = plt.subplots(num_cf)
    plt.figure()
    for x in range(num_cf):
        axs2[x].plot(range(0,100), pos_data[x])  
        axs2[x].set_title(str(x+1))
    plt.show()

def motion(v):
    for i in range(num_cf):
        allcfs.crazyflies[i].cmdVelocityWorld([v[3*i],v[3*i+1],v[3*i+2]], yawRate=0)
    timeHelper.sleep(0.01)

if __name__ == "__main__":
    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    i = 0
    while i < 100:
        # goal = goalMotion(totalTime=[2.146,-0.73, 1, -1.67], radius=[4,2,3,3.4], time_i=i)
        goal = goalMotion(totalTime = [8.146, -7.73], radius = [4,2], time_i = i)
        v = dontCrash(goal, time_i = i)
        motion(v)
        i+=1 
    plot()
