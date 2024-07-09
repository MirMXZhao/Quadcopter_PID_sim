#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *


Z = 1.0
sleepRate = 30


def goCircle(timeHelper, allcfs, totalTime, radius, kPosition):
    # each crazyflie flies in a circle for 10 seconds
    for cf in allcfs.crazyflies:
        print(cf)
        startTime = timeHelper.time()
        pos = cf.position()
        startPos = cf.initialPosition + np.array([0, 0, Z])
        center_circle = startPos - np.array([radius, 0, 0])
        time = 0
        while time < 4*np.pi:
            time = timeHelper.time() - startTime
            omega = 2 * np.pi / totalTime
            vx = -radius * omega * np.sin(omega * time)  
            vy = radius * omega * np.cos(omega * time)
            desiredPos = center_circle + radius * np.array(
                [np.cos(omega * time), np.sin(omega * time), 0])
            errorX = desiredPos - cf.position() 
            cf.cmdVelocityWorld(np.array([vx, vy, 0] + kPosition * errorX), yawRate=0) #kPosition*errorX adjusts vel to acc for differences bw expectations vs reality
            timeHelper.sleepForRate(sleepRate)
        cf.cmdVelocityWorld([0,0,0], 0)
        # cf.goTo(startPos, 0, 1.0)
        # how do we make the crazyflie stop after it's done and not fly in a random direction?

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=Z, duration=2.0+Z)
    timeHelper.sleep(2 + Z)

    goCircle(timeHelper, allcfs, totalTime=4, radius=1, kPosition=1)
    
