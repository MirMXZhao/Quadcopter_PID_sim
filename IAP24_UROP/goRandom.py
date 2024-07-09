#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
from qpsolvers import solve_qp
import random 
import math

Z = 1.0
sleepRate = 30


def goRandom(allcfs): #random flight
    numCf= len(allcfs.crazyflies)
    v = [0]*3
    while True:
        for i in range(numCf):
            v[0] = random.uniform(-1,1)
            v[1] = random.uniform(-1,1)
            v[2] = random.uniform(-0.8,1)
            for j in range(3): #ensures they stay more or less restricted to a visible zone in the simulation
                if(allcfs.crazyflies[i].position()[j]>=4):
                    v[j] = -0.8
                if(allcfs.crazyflies[i].position()[j]<=0):
                    v[j] = 0.8
            allcfs.crazyflies[i].cmdVelocityWorld(np.array([v[0], v[1], v[2]]), yawRate=0) # causes actual motion
            timeHelper.sleep(1)

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    goRandom(allcfs)
