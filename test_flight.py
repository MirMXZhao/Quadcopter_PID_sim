#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *

Z = 1.0

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    cf = allcfs.crazyflies[0]

    # print(cf.position())
    # cf.takeoff(targetHeight=Z, duration=1.0+Z)
    # timeHelper.sleep(1+Z)
    # print("right before go to:", + cf.position())
    # pos1 = np.array([cf.position()[0], cf.position()[1],Z]) + np.array([-0.5, -1, 0])
    # print("intended go to:", + pos1)
    # cf.goTo(pos1, 0, 3)
    # timeHelper.sleep(2+Z)
    # print("after go to:", + cf.position())
    # cf.cmdVelocityWorld([0, 0.3, 0], yawRate=0)
    # timeHelper.sleep(4)
    # print("after velocity command:", + cf.position())
    # cf.cmdVelocityWorld([0, 0, -0.1], yawRate=0)
    # timeHelper.sleep(4)

    print(cf.position())
    cf.takeoff(targetHeight=Z, duration=1.0+Z)
    timeHelper.sleep(1+Z)
    cf.cmdVelocityWorld([0, -0.6, 0], yawRate=0)
    timeHelper.sleep(6)
    print("after velocity command:", + cf.position())
    cf.cmdVelocityWorld([0, 0, -0.5], yawRate=0)
    timeHelper.sleep(7)

    # cf.land(targetHeight=0.02, duration=2.5)
    # timeHelper.sleep(1.5+Z)
    # print("landed")
    # print(cf.position())
