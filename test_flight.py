"""
- when running, set --maxvel 2
"""
#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
import math
import matplotlib.pyplot as plt

crazyflies_yaml = """
crazyflies:
- id: 1
  channel: 110
  initialPosition: [0, 0.0, 0.0]
"""

swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml)
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs
cf = allcfs.crazyflies[0] #There should only be one crazyflie for this program
sleepRate = 30
num_cf = len(allcfs.crazyflies)
Z = 1.0

#which test to run 
test_num = 5

def plot3D_both(toplot1, toplot2):
   """
   plot both
   """
   fig = plt.figure()
   ax = plt.axes(projection='3d')
   x_val1 = [vector[0] for vector in toplot1]
   y_val1 = [vector[1] for vector in toplot1]
   z_val1 = [vector[2] for vector in toplot1]
   x_val2 = [vector[0] for vector in toplot2]
   y_val2 = [vector[1] for vector in toplot2]
   z_val2 = [vector[2] for vector in toplot2]
   ax.plot3D(x_val1, y_val1, z_val1, 'blue', label = "real")
   ax.plot3D(x_val2, y_val2, z_val2, 'orange', label = "desired")

if __name__ == "__main__":
    if test_num == 1: 
        # testing cmdFullState
        swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml)
        for i in range(5):
            cf.cmdFullState([0, 0, 0], [0, 0, 0.5], [0, 0, 0.5], 0, [0,0,0])
            timeHelper.sleep(0.05)
            cf.cmdFullState([0, 0, 1], [0, 0, -0.5], [0, 0, -0.5], 0, [0,0,0])
            timeHelper.sleep(0.05)
        #confusing!! it is not doing what im telling it to do 
    elif test_num == 2: 
        cf.takeoff(targetHeight=Z, duration=1.0+Z)
        timeHelper.sleep(1+Z)
        cf.cmdVelocityWorld([0, -0.6, 0], yawRate=0)
        timeHelper.sleep(6)
        print("after velocity command:", + cf.position())
        cf.cmdVelocityWorld([0, 0, -0.5], yawRate=0)
        timeHelper.sleep(7)
    elif test_num == 3: 
        cf.cmdVelocityWorld([0,0,0.1], 0) 
        timeHelper.sleep(0.5)
        cf.cmdVelocityWorld([0,0,0.2], 0) 
        timeHelper.sleep(0.5)
        cf.cmdVelocityWorld([0,0,0.1], 0) 
        timeHelper.sleep(0.5)
        cf.cmdVelocityWorld([0,0,0.1], 0) 
        timeHelper.sleep(2)
        cf.cmdVelocityWorld([0,0,0.1], 0) 
        timeHelper.sleep(0.5)
        cf.cmdVelocityWorld([0,0,0.1], 0) 
        timeHelper.sleep(0.5)
    elif test_num == 4:
        """
        testing just the circular motion 
        """
        state = []
        r = 4
        speed = 0.01
        height = 0
        step = 0.003
        num_iter = 50
        real_states = [] 
        for i in range(num_iter+2):
            theta = i*speed
            velx = -r * math.sin(theta) * speed/step 
            vely = r* math.cos(theta) * speed/step 
            accx = - r* math.cos(theta) * speed*speed/(step*step) 
            accy = -r * math.sin(theta) * speed/step 
            state.append([r*math.cos(theta), r*math.sin(theta), height, velx, vely, 0, accx, accy, 0])
        for i in range(num_iter):
            cf.cmdFullState(state[i][0:3], state[i][3:6], state[i][6:9], 0, [0,0,0])
            timeHelper.sleep(0.003)
            real_states.append(cf.position())
        plot3D_both(real_states, state)
        plt.show()
    elif test_num == 5: 
        for _ in range(100):
            cf.cmdFullState([0,0,0], [0,0,0], [0,0,0], 0, [0,0,0])
            timeHelper.sleep(0.003)
            print(cf.position())


