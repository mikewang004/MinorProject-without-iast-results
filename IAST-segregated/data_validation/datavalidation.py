#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp

temp = 400
no_molecules = 3

path = "../automated_output/%d_molecules/%dK_temperature" %(no_molecules, temp)


#data1 = np.loadtxt(path + "/2mC6-3mC6-33mC5-C7/0.100000-0.650000-0.050000-0.200000.txt", skiprows=1)
#data2 = np.loadtxt(path + "/22mC5-2mC6-33mC5-23mC5/0.100000-0.650000-0.050000-0.200000.txt", skiprows=1)
data1 = np.loadtxt(path + "/2mC6-3mC6-33mC5/0.200000-0.300000-0.500000.txt", skiprows=1)
data2 = np.loadtxt(path + "/22mC5-33mC5-C7/0.200000-0.300000-0.500000.txt", skiprows=1)


for i in range(1, no_molecules+1):
    plt.loglog(data1[:, 0], data1[:, i], "o", label="%d-th molecule" %(i))
    plt.loglog(data2[:, 0], data2[:, i], ".", label="%d-th molecule" %(i))
plt.legend()
plt.show()