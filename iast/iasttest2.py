#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 

input_path = "../Raspa/outputs/"

def return_langmuir(df_iso):
    return pyiast.ModelIsotherm(df_iso, loading_key = "molkg", pressure_key = "pressure", model = "DSLangmuir")

molecule_1 = r"$CH_7$"
molecule_2 = r"$3mC6$"
temp = r"$500 K$"

moleculeiso_1 = return_langmuir(pd.read_csv(input_path + "C7/C7-500out.txt"))
moleculeiso_2 = return_langmuir(pd.read_csv(input_path + "3mC6/3mC6-500out.txt"))

print(moleculeiso_1.params)
moleculeiso_2.print_params()



#Plot pure isotherms 

""" no_fracs = 2
no_pressures = 50
#gas_frac = np.linspace(1.0, 0.5, no_fracs)
partial_pressures = np.logspace(0, 8, num=no_pressures)
#pyiast.plot_isotherm(moleculeiso_1)
plt.semilogx(partial_pressures, moleculeiso_1.loading(partial_pressures), "ro", label=molecule_1 + r", homogeneous gas")
plt.semilogx(partial_pressures, moleculeiso_2.loading(partial_pressures), "go", label=molecule_2 + r", homogeneous gas")
plt.title("Loadings of a " + molecule_1 + " and " + molecule_2 +  " mixture at temperature " + temp)
#plt.title("Loadings of a " + molecule_1 + " mixture at temperature " + temp)
plt.xlabel("Pressure (bar)")
plt.ylabel("Loading (mol/kg)")
plt.legend()
plt.show() """

