#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp

data = np.loadtxt("fort.25")

for i in range(0, 10):
    plt.loglog(data[:, 0], data[:, 1], label="component 1")
    plt.loglog(data[:, 0], data[:, 2], label="component 2 ")

#plt.legend()
plt.show()