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
safeRadius = 0.5

swarm = Crazyswarm()
allcfs = swarm.allcfs
num_cf = len(allcfs.crazyflies)
data = np.zeros((num_cf, num_cf, 100))

walls = [5, -5, 5, -5]
wallsradius = 0.1

def goCircle(timeHelper, totalTime, radius, kPosition, time_i):
    startTime = timeHelper.time()
    numCf= len(allcfs.crazyflies)
    pos = [None]*numCf
    startPos = [None]*numCf
    center_circle = [None]*numCf
    v_cmd = np.zeros((numCf, 3))

    for i in range(numCf):
        pos[i] = allcfs.crazyflies[i].position()
        startPos[i] = allcfs.crazyflies[i].initialPosition + np.array([0, 0, Z])
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    # time = timeHelper.time() - startTime
    time = time_i
    for i in range(numCf):
        omega = 0.5* np.pi / totalTime[i]
        vx = -radius[i] * omega * np.sin(omega * time)  
        vy = radius[i] * omega * np.cos(omega * time)
        v_cmd[i, :] = np.array([vx, vy, 0])
        timeHelper.sleepForRate(sleepRate)
    v = dontCrash(v_cmd, safeRadius, time_i)
    return v

def dontCrash(v_cmd, sradius, time_i):
    pos = np.zeros((num_cf, 3))
    i = 0
    for cf in allcfs.crazyflies:
        pos[i, :] = cf.position()
        i = i+1

    P = np.identity(3*num_cf)

    q = np.zeros((num_cf)*3)
    for i in range(num_cf):
        q[i*3] = -v_cmd[i][0]
        q[i*3 +1] = -v_cmd[i][1]
        q[i*3 +2] = -v_cmd[i][2]
        
    Pos = np.zeros((num_cf, 3))
    i = 0
    for cf in allcfs.crazyflies:
        Pos[i, :] = cf.position()
        i = i+1

    dim = int(num_cf*(num_cf-1)/2) 
    G = np.zeros((num_cf*4, num_cf*3))
    x = 0
    for i in range(num_cf): # describes condition 3 (avoiding walls)
        for j in range(4):
            if j < 2:
                G[x][3*i+1] = (walls[j]- pos[i][1]) 
            else:
                G[x][3*i] =  (walls[j]- pos[i][0])
            x+=1 

    h = np.zeros(num_cf*4)
    x=0
    for i in range(num_cf):
        for j in range(4):
            if j<2:
                h[x] = - (wallsradius*wallsradius - (pos[i][1]-walls[j])*(pos[i][1]-walls[j]))/2 
            else:
                h[x] = - (wallsradius*wallsradius - (pos[i][0]-walls[j])*(pos[i][0]-walls[j]))/2
            x+=1
    

    v = solve_qp(P,q,G,h,A=None,b=None, solver = "daqp")
    # print("v_cmd", + v_cmd)
    # print("pos", + Pos)
    # print("q", + q)
    # print("G", + G)
    # print("h", + h)
    # print("v", + v)
    # print()
    # motion(v)
    return v

def plot(data, num_cf):
    i = 211
    plt.figure()
    for x in range(num_cf):
        for y in range(x+1, num_cf):
            plt.subplot()
            plt.plot(range(0,100), data[x][y])
            i +=1
        
    # plt.show()

    fig, axs = plt.subplots(num_cf-1, num_cf-1, squeeze = False) #setting squeeze to false allows us to have only 1 graph
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
    timeHelper = swarm.timeHelper
    
    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    i = 0
    while i < 100:
        v = goCircle(timeHelper, totalTime=[2.146,-0.73, 1, -1.67], radius=[4,2,3,3.4], kPosition=1, time_i=i)
        motion(v)
        i+=1
    
    plot(data, num_cf)
