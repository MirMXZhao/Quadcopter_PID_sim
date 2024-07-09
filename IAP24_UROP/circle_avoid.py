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


def goRandom(allcfs): #random flight
    numCf= len(allcfs.crazyflies)
    v_cmd = np.zeros((numCf, 3))
    v = [0]*3
    for i in range(numCf):
        v[0] = random.uniform(-0.5,0.5)
        v[1] = random.uniform(-0.5,0.5)
        v[2] = random.uniform(-0.5,0.5)
        for j in range(3): #ensures they stay more or less restricted to a visible zone in the simulation
            if(allcfs.crazyflies[i].position()[j]>=4):
                v[j] = -0.6
            if(allcfs.crazyflies[i].position()[j]<=-1.5):
                v[j] = 0.6
        v_cmd[i, :] = np.array([v[0], v[1], v[2]])
        # allcfs.crazyflies[i].cmdVelocityWorld(np.array([v[0], v[1], v[2]]), yawRate=0) # causes actual motion
    v = dontCrash(v_cmd, allcfs, safeRadius)
    return v
    

def goCircle(timeHelper, allcfs, totalTime, radius, kPosition, time_i):
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
    v = dontCrash(v_cmd, allcfs, 1)
    return v

def dontCrash(v_cmd, allcfs, sradius):
    num_cf = len(allcfs.crazyflies)

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
    G = np.zeros((dim, num_cf*3))
    x = 0
    for i in range(num_cf):
        for j in range(i+1, num_cf):
            dist_ij = np.linalg.norm(Pos[i] - Pos[j])
            if dist_ij > 2 * sradius:
                x = x+1
                continue
            G[x][3*j] = Pos[i][0] - Pos[j][0]
            G[x][3*j+1] = Pos[i][1] - Pos[j][1]
            G[x][3*j+2] = Pos[i][2] - Pos[j][2]
            G[x][3*i] = -Pos[i][0] + Pos[j][0]
            G[x][3*i+1] = -Pos[i][1] + Pos[j][1]
            G[x][3*i+2] = -Pos[i][2] + Pos[j][2]
            x+=1

    h = np.zeros(dim)
    x=0
    for i in range(num_cf):
        for j in range(i+1, num_cf):
            h[x] = (sradius*sradius - math.dist(Pos[i], Pos[j])*math.dist(Pos[i], Pos[j]))/2
            x+=1

    v = solve_qp(P,q,G,h,A=None,b=None, solver = "daqp")
    
    # print("v_cmd", + v_cmd)
    # print("pos", + Pos)
    # print("q", + q)
    # print("G", + G)
    # print("h", + h)
    # print("v", + v)
    # print()
    # motion(v, allcfs)

    return v

def plot(data, num_cf):
    total_plots = int(num_cf*(num_cf -1)/2)
    # f = np.empty(total_plots)
    # ax = np.empty(total_plots)

    # for i in range(total_plots):
    #     f[i] = plt.figure()
    #     ax[i] = f[i].add_subplot(111)
    
    i = 211
    plt.figure()
    for x in range(num_cf):
        for y in range(x+1, num_cf):
            plt.subplot()
            plt.title("Distance between "+str(x)+ " and " + str(y))
            plt.plot(range(0,100), data[x][y])
            i +=1
        
    # plt.show()

    fig, axs = plt.subplots(num_cf-1, num_cf-1)
    plt.figure()
    for x in range(num_cf):
        for y in range(x+1, num_cf):
            axs[x,y-1].plot(range(0,100), data[x][y])
            axs[x,y-1].set_title(str(x+1)+ " and " + str(y+1))
        
    plt.show()


def motion(v, allcfs):
    for i in range(len(allcfs.crazyflies)):
        allcfs.crazyflies[i].cmdVelocityWorld([v[3*i],v[3*i+1],v[3*i+2]], yawRate=0) # change this so it fits with the definition of v
    timeHelper.sleep(0.1)

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    num_cf = len(allcfs.crazyflies)

    data = np.zeros((num_cf, num_cf, 100))

    i = 0
    while i < 100:
        v = goCircle(timeHelper, allcfs, totalTime=[2.146,-0.73, 1, -1.67], radius=[4,2,3,3.4], kPosition=1, time_i=i)
        motion(v, allcfs)

        for x in range(num_cf):
            for y in range(x+1, num_cf):
                data[x][y][i] = math.dist(allcfs.crazyflies[x].position(), allcfs.crazyflies[y].position())
        i +=1

    # f1 = plt.figure()
    # f2 = plt.figure()
    # ax1 = f1.add_subplot(111)
    # ax1.plot(data[0][1])
    # ax2 = f2.add_subplot(111)
    # ax2.plot(data[0][2])
    # plt.show()
    
    plot(data, num_cf)
