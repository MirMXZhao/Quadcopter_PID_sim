import numpy as np
from pycrazyswarm import *

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0
Z = 1.0

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=Z, duration=1.0+Z)
    timeHelper.sleep(1.5+Z)
    for i in range(2):
        pos = np.array(allcfs.crazyflies[i].initialPosition) + np.array([1, 0, Z])
        allcfs.crazyflies[i].goTo(pos, 0, 1.0)
        print("hi")
    
    timeHelper.sleep(1.0+Z)

    allcfs.land(targetHeight=0.02, duration=1.0+Z)
    timeHelper.sleep(1.0+Z)

    # for cf in allcfs.crazyflies:
    #     cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    #     timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    #     cf.land(targetHeight=0.04, duration=2.5)
    #     timeHelper.sleep(TAKEOFF_DURATION)

if __name__ == "__main__":
    main()
