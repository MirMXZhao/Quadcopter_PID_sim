#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *


Z = 1.0
sleepRate = 30

# for fun: work on making them fly at the same tangentia velocity (not angular)

def goCircle(timeHelper, allcfs, totalTime, radius, kPosition):
    startTime = timeHelper.time()
    numCf= len(allcfs.crazyflies)
    pos = [None]*numCf
    startPos = [None]*numCf
    center_circle = [None]*numCf

    for i in range(numCf):
        pos[i] = allcfs.crazyflies[i].position()
        startPos[i] = allcfs.crazyflies[i].initialPosition + np.array([0, 0, Z])
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    while True:
        time = timeHelper.time() - startTime
        omega = (10 * np.pi / totalTime)/(radius[i]*radius[i])
        for i in range(numCf):
            vx = -radius[i] * omega * np.sin(omega * time)  
            vy = radius[i] * omega * np.cos(omega * time)
            desiredPos = center_circle[i] + radius[i] * np.array([np.cos(omega * time), np.sin(omega * time), 0])
            errorX = desiredPos - allcfs.crazyflies[i].position() 
            allcfs.crazyflies[i].cmdVelocityWorld(np.array([vx, vy, 0] + kPosition * errorX), yawRate=0)
            timeHelper.sleepForRate(sleepRate)

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    goCircle(timeHelper, allcfs, totalTime=4, radius=[1,2,3,4], kPosition=1)
    
