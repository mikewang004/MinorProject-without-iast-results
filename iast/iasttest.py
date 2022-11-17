#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:52:33 2022

@author: mike
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 

df_ch4 = pd.read_csv("/home/mike/Documents/uni/alice/Final_project/MinorProject/iast/ethaan_iso.txt")
df_ch3ch3 = pd.read_csv("/home/mike/Documents/uni/alice/Final_project/MinorProject/iast/methane_iso2.txt")

# Fit Langmuir isotherm to dataframe 

ch4_isotherm = pyiast.ModelIsotherm(df_ch4, loading_key = "molkg", pressure_key = "pressure", model = "Langmuir")
ch3ch3_isotherm = pyiast.ModelIsotherm(df_ch3ch3, loading_key = "molkg", pressure_key = "pressure", model = "Langmuir")

#pyiast.plot_isotherm(ch3ch3_isotherm)
#pyiast.plot_isotherm(ch4_isotherm)

ch4_isotherm.print_params()
ch3ch3_isotherm.print_params()

gas_frac = np.array([0.05, 0.95])
total_pressure = np.linspace(0, 10e4, 50)
mixture = np.zeros([50, 2])
for i in range(0,50):
    mixture[i] = pyiast.iast(total_pressure[i] * gas_frac, [ch4_isotherm, ch3ch3_isotherm])

plt.plot(total_pressure, mixture[:, 0], color="red")
plt.plot(total_pressure, mixture[:, 1], color="green")
plt.title("Loading of a %f - %f % methane -ethane mixture",)
plt.xlabel("Pressure (bar)")
plt.ylabel("Loading (mol/kg)")
plt.show()