#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *
from qpsolvers import solve_qp
import random 
import math
import matplotlib.pyplot as plt
# from IPython import get_ipython

if __name__ == "__main__":
    # testing matplotplib
    # mylist = list((1,2,3,100))
    # # get_ipython.run_line_magic('matplotlib', 'inline')
    # plt.plot(range(1,5), mylist)
    # plt.show()

    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot((1,5,2,7))
    ax2 = f2.add_subplot(111)
    ax2.plot(range(4,14))
    plt.show()


    
