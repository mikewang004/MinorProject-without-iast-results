#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 
import subprocess

#importpath1 = "C7-500_3mC6-500/"
importpath1 = "C7-600_22mC5-600/"
importpath2 = "22mC5-500_23mC5-600/"
#Note column 1 is C7; column 2 3mC6. 
#data1 = np.loadtxt(importpath1 + "C7-500-0.02_3mC6-500-0.98.txt", skiprows=1)
#data2 = np.loadtxt(importpath1 + "C7-500-0.84_22mC5-500-0.16.txt", skiprows=1)
data3 = np.loadtxt(importpath1 + "C7-600-0.51_22mC5-600-0.49.txt", skiprows=1)

#data1 = np.loadtxt(importpath2 + "22mC5-500-0.02_23mC5-500-0.98.txt", skiprows=1)
#data2 = np.loadtxt(importpath2 + "22mC5-500-0.84_23mC5-500-0.16.txt", skiprows=1)
#data3 = np.loadtxt(importpath2 + "22mC5-500-0.51_23mC5-500-0.49.txt", skiprows=1)

#plt.loglog(data1[:, 0], data1[:, 2], "b.")
#plt.loglog(data1[:, 0], data1[:, 1], "g.")
#plt.loglog(data2[:, 0], data2[:, 2], "y.")
#plt.loglog(data2[:, 0], data2[:, 1], "g.")
plt.loglog(data3[:, 0], data3[:, 2], "r.")
plt.loglog(data3[:, 0], data3[:, 1], "g.")
plt.show()