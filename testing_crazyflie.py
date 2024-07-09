"""
testing out cmdFullState
Should only be run with one crazyflie
"""
import numpy as np
from pycrazyswarm import *

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0
Z = 1.0
num_rounds = 300

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs
num_cf = len(allcfs.crazyflies)
startPos = [None]*num_cf
sleepRate = 20

cxdata = np.zeros((num_cf, num_rounds))
cydata = np.zeros((num_cf, num_rounds))
czdata = np.zeros((num_cf, num_rounds))

#uneven circular motion for goals
def goalMotion(totalTime, radius,  time_i):
    startTime = timeHelper.time()
    pos = [None]*num_cf
    center_circle = [None]*num_cf
    goal = np.zeros((2*num_cf, 3))

    for i in range(num_cf):
        pos[i] = allcfs.crazyflies[i].position()
        center_circle[i] = startPos[i] - np.array([radius[i], 0, 0])

    # time = timeHelper.time() - startTime
    time = time_i
    for i in range(num_cf):
        omega = 0.5* np.pi / totalTime[i]
        vx = -radius[i] * omega * np.sin(omega * (time))  
        vy = radius[i] * omega * np.cos(omega * (time)) #CHANGED from time to time - speed >> worked????
        velocity = np.array([vx, vy, 0])
        position = center_circle[i] + radius[i] * np.array([np.cos(omega * time), np.sin(omega * time), 0])
        acceleration = 0.1*(position- center_circle[i])
        yaw = 0
        omega = np.array([0,0,0])
        allcfs.crazyflies[i].cmdFullState(position, velocity, acceleration, yaw, omega)

        # cxdata[time_i] = goal[2*i+1][0] # for plotting
        # cydata[time_i] = goal[2*i+1][1]
        # czdata[time_i] = goal[2*i+1][2]

        timeHelper.sleepForRate(sleepRate)

    return goal #goal encodes desired position in odd indices and desired velocity in even indices

def main():
    allcfs.takeoff(targetHeight=Z, duration=1.0+Z)
    timeHelper.sleep(1.5+Z)

    for i in range(num_cf):
        startPos[i] = allcfs.crazyflies[i].position()
        
    goalMotion([4,4,4,4], [1,1,1,1], 3)
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
