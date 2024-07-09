# testing collision avoidance and goal reachin for 2 drones. no walls, animation currently commented out

#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
from qpsolvers import solve_qp
import random 
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Z = 1.0
sleepRate = 20
sradius = 1

num_rounds = 200 # determines how long the program runs for 

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs

num_cf = len(allcfs.crazyflies)
startPos = [None]*num_cf

actual_data = np.zeros((num_cf, num_cf, num_rounds))
pos_data = np.zeros((num_cf, num_rounds))

xdata = np.zeros((num_cf, num_rounds))
ydata = np.zeros((num_cf, num_rounds))
zdata = np.zeros((num_cf, num_rounds))

cxdata = np.zeros((num_cf, num_rounds)) #supposed position
cydata = np.zeros((num_cf, num_rounds))
czdata = np.zeros((num_cf, num_rounds))

# animx = [[]]*num_cf
# animy = [[]]*num_cf
# animz = [[]]*num_cf
# canimx = [[]]*num_cf #animation for intended path
# canimy = [[]]*num_cf
# canimz = [[]]*num_cf
animx1 = []
animy1 = []
animz1 = []
animx2 = []
animy2 = []
animz2 = []
canimx1 = [] #animation for intended path
canimy1 = []
canimz1 = []
canimx2 = [] #animation for intended path
canimy2 = []
canimz2 = []
fig4 = plt.figure()
ax1 = plt.axes(projection='3d')


#uneven circular motion for goals
def goalMotion(totalTime, radius, time_i):
    startTime = timeHelper.time()
    pos = [None]*num_cf
    center_circle = [None]*num_cf
    goal = np.zeros((2*num_cf, 3))

    for i in range(num_cf):
        pos[i] = allcfs.crazyflies[i].position()
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    # time = timeHelper.time() - startTime
    speed = 0.05 #makes it go slower
    time = time_i*speed
    for i in range(num_cf):
        omega = 0.5* np.pi / totalTime[i]
        vx = -radius[i] * omega * np.sin(omega * (time + speed))  
        vy = radius[i] * omega * np.cos(omega * (time + speed))
        goal[2*i, :] = np.array([vx, vy, 0])
        goal[2*i+1, :] = center_circle[i] + radius[i] * np.array([np.cos(omega * time), np.sin(omega * time), 0])
        
        cxdata[i][time_i] = goal[2*i+1][0] # for plotting
        cydata[i][time_i] = goal[2*i+1][1]
        czdata[i][time_i] = goal[2*i+1][2]

        timeHelper.sleepForRate(sleepRate)

    return goal #goal encodes desired position in odd indices and desired velocity in even indices

# problem: minimize v^2 given 
#1. (v - v_goal)^(T)(p - p_goal) <= -1/2(p - p_goal)^(T)(p-p_goal) +1/2*0.01^2
#2. (vi-vj)^(T)(pi-pj) <= 1/2(|pi-pj|^2 - d^2)
# also keeps track of data
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
    G = np.zeros((dim + num_cf, num_cf*3))
    x = 0
    for i in range(num_cf): # This describes condition 1 (getting to the goal position)
        G[x][3*i] = pos[i][0] - goal[2*i+1][0]
        G[x][3*i+1] = pos[i][1] - goal[2*i+1][1]
        G[x][3*i+2] = pos[i][2] - goal[2*i+1][2]
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
        h[x] = -math.dist(pos[i], goal[2*i+1])*math.dist(pos[i], goal[2*i+1])/2 +0.01*0.01/2 + np.dot(pos[i] - goal[2*i+1], goal[2*i])
        x+=1
    for i in range(num_cf):
        for j in range(i+1, num_cf): #bound for condition 2
            h[x] = - (sradius*sradius - math.dist(pos[i], pos[j])*math.dist(pos[i], pos[j]))/2
            actual_data[i][j][time_i] = math.dist(pos[i], pos[j]) #collects data for plotting later
            x+=1

    for i in range(num_cf): #takes position data for future plotting
        pos_data[i][time_i] = math.dist(pos[i], goal[2*i+1])
    
    for i in range(num_cf): #takes position data for future plotting
        xdata[i][time_i] = pos[i][0]
        ydata[i][time_i] = pos[i][1]
        zdata[i][time_i] = pos[i][2]

    v = solve_qp(P,q,G,h,A=None,b=None, solver = "daqp") 
    v = np.clip(v, -2, 2)
    # print("goal ", + goal)
    # print("pos", + pos)
    # print("G", + G)
    # print("h", + h)
    # print("v", + v)

    return v

def plot():
    #graphs pairwise distance between drones
    fig, axs = plt.subplots(num_cf-1, num_cf-1, squeeze = False) # setting squeeze value to false forces it to work for 2 drones
    plt.figure()
    for x in range(num_cf): 
        for y in range(x+1, num_cf):
            axs[x,y-1].plot(range(0,num_rounds), actual_data[x][y]) 
            axs[x,y-1].plot(range(0,num_rounds), [sradius]*num_rounds)
            axs[x,y-1].set_title(str(x+1)+ " and " + str(y+1)) 

    #graphs distance between drone and goal position 
    fig2, axs2 = plt.subplots(num_cf)
    plt.figure()
    for x in range(num_cf):
        axs2[x].plot(range(0,num_rounds), pos_data[x])  
        axs2[x].set_title(str(x+1))
    
    #shows 3d flight path
    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([-2, 3])
    ax.set_ylim([-2, 3])
    ax.set_zlim([0, 2])
    for i in range(num_cf):
        ax.plot3D(xdata[i], ydata[i], zdata[i], 'gray')
        ax.plot3D(cxdata[i], cydata[i], czdata[i], 'orange')
        ax.plot3D(xdata[1], ydata[1], zdata[1], 'gray')
    
    # #animation for 3d flight path
    # ax1.set_xlim([-2, 3])
    # ax1.set_ylim([-2, 3])
    # ax1.set_zlim([0, 2])

    # anim = FuncAnimation(fig4, animate, frames = num_rounds, repeat = True, interval = 20)
    plt.show()
    
# def animate(i):
#     animx1.append(xdata[0][i]) #change this so it isn't hard coded 
#     animy1.append(ydata[0][i])
#     animz1.append(zdata[0][i])
#     animx2.append(xdata[1][i])
#     animy2.append(ydata[1][i])
#     animz2.append(zdata[1][i])
    
#     ax1.plot3D(animx1, animy1, animz1, color = "blue")
#     ax1.plot3D(animx2, animy2, animz2, color = "red")
    
#     canimx1.append(cxdata[0][i])
#     canimy1.append(cydata[0][i])
#     canimz1.append(czdata[0][i])
#     canimx2.append(cxdata[1][i])
#     canimy2.append(cydata[1][i])
#     canimz2.append(czdata[1][i])
    
#     ax1.plot3D(canimx1, canimy1, canimz1, color = "orange")
#     ax1.plot3D(canimx2, canimy2, canimz2, color = "purple")

#     # for j in range(num_cf): # has errors where all the drone lines are plotted together
#     #     animx[j].append(xdata[j][i])
#     #     animy[j].append(ydata[j][i])
#     #     animz[j].append(zdata[j][i])
    
#     # ax1.plot3D(animx[0], animy[0], animz[0], color = "blue")
#     # ax1.plot3D(animx[1], animy[1], animz[1], color = "red")
    
#     # for j in range(num_cf):
#     #     canimx[j].append(cxdata[j][i])
#     #     canimy[j].append(cydata[j][i])
#     #     canimz[j].append(czdata[j][i])
    
#     # ax1.plot3D(canimx[j], canimy[j], canimz[j], color = "orange")
#     # ax1.plot3D(canimx[j], canimy[j], canimz[j], color = "purple")


def motion(v):
    for i in range(len(allcfs.crazyflies)):
        allcfs.crazyflies[i].cmdVelocityWorld([v[3*i]*0.3,v[3*i+1]*0.3,v[3*i+2]*0.05], yawRate=0) # change this so it fits with the definition of v
    timeHelper.sleep(0.02)

if __name__ == "__main__":
    try:
        allcfs.takeoff(targetHeight=Z, duration=1+Z)
        timeHelper.sleep(1 + Z + 2.5)

        for i in range(num_cf):
            startPos[i] = allcfs.crazyflies[i].position()
            # print(startPos[i])

        # totalTime = np.random.uniform(low = -2, high = 2, size = (num_cf,))
        # radius = np.random.uniform(low = 1.5, high = 5, size = (num_cf,))
        i = 0
        while i < num_rounds:
            # goal = goalMotion(totalTime, radius, time_i = i) #for a variable number of drones
            # goal = goalMotion(totalTime=[2.146,-0.73, 1, -1.67], radius=[4,2,3,3.4], time_i=i) #for 4 drones
            goal = goalMotion(totalTime = [6.9, -2.4], radius = [0.7, 0.89], time_i =i)
            v = dontCrash(goal, time_i = i)
            motion(v)
            i+=1
        
        for i in range(num_cf):
            allcfs.crazyflies[i].notifySetpointsStop() #changes to high level commands
        
        allcfs.land(targetHeight=0.04, duration=2.5)
        timeHelper.sleep(Z+1)

        # for i in range(num_cf):
        #     print(xdata[i])
        #     print(ydata[i])
        #     print(zdata[i])
        plot()
    except KeyboardInterrupt: #forces land if cancelled
        print("oops")
        for i in range(num_cf):
            allcfs.crazyflies[i].notifySetpointsStop() #changes to high level commands

        allcfs.land(targetHeight=0.04, duration=2.5)
        timeHelper.sleep(Z+1)