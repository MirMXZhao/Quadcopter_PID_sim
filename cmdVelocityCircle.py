#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *


Z = 1.0
sleepRate = 30
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0

def goCircle(timeHelper, cf, totalTime, radius, kPosition):
        startTime = timeHelper.time()
        pos = cf.position()
        startPos = cf.initialPosition + np.array([0, 0, Z])
        center_circle = startPos - np.array([radius, 0, 0])
        time = 0
        while time < 10:
            time = timeHelper.time() - startTime
            omega = 2 * np.pi / totalTime
            vx = -radius * omega * np.sin(omega * time)  
            vy = radius * omega * np.cos(omega * time)
            desiredPos = center_circle + radius * np.array([np.cos(omega * time), np.sin(omega * time), 0])
            # errorX = desiredPos - cf.position() 
            # cf.cmdVelocityWorld(np.array([vx, vy, 0] + kPosition * errorX), yawRate=0)
            cf.cmdVelocityWorld(np.array([vx, vy, 0]), yawRate=0)
            timeHelper.sleepForRate(sleepRate)
            print("actual ",+cf.position())
            print("desired ",+ desiredPos)


if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION +1)
    # swarm = Crazyswarm()
    # timeHelper = swarm.timeHelper
    # allcfs = swarm.allcfs
    # cf = allcfs.crazyflies[0]
    # print("starting")
    # cf.takeoff(targetHeight=Z, duration=1.0+Z)
    # timeHelper.sleep(2 + Z)

    print("takeoff")
    goCircle(timeHelper, cf, totalTime=6, radius=0.7, kPosition=1)
    cf.cmdVelocityWorld([0, 0, -1], yawRate=0)
    timeHelper.sleep(6)
    # cf.land(targetHeight=0.04, duration=2.5) #added to test in real life
    # timeHelper.sleep(5+Z)
