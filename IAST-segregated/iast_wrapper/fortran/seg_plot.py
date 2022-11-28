#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp

data = np.loadtxt("fort.25")


plt.loglog(data[:, 0], data[:, 1], "go", label="component 1")
plt.loglog(data[:, 0], data[:, 2], "ro", label="component 2 ")

plt.xlabel("Pressure (pascal)")
plt.ylabel("Loading (mol/kg)")
plt.legend()
plt.show()