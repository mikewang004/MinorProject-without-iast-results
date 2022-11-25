#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 

startnum, stopnum = 0,0
input_path = "../../Raspa/outputs/"

def return_langmuir(df_iso):
    return pyiast.ModelIsotherm(df_iso, loading_key = "molkg", pressure_key = "pressure", model = "DSLangmuir")

molecule_1 = r"$CH_7$"
molecule_2 = r"$3mC6$"
temp = r"$400 K$"
no_components = 2 
moleculeiso_1 = return_langmuir(pd.read_csv(input_path + "C7/C7-500out.txt"))
moleculeiso_2 = return_langmuir(pd.read_csv(input_path + "3mC6/3mC6-500out.txt"))

moleculeiso_1.print_params()
moleculeiso_2.print_params()

#Write params to fortran-file 
startline, stopline = 'C     Start for Python', 'C     End for Python'
with open("fortran/testiast.f", "r+") as file:
    data = file.readlines()

for num, line in enumerate(data, 1):
    if startline in line:
        startnum = num
    elif stopline in line:
        stopnum = num

print(startnum, stopnum)
del data[startnum:stopnum-1]
print(data)

for i in range(startnum + 1, startnum + 11 * no_components):
    

#Read and plot output fortran-file 

output = np.loadtxt("fortran/fort.25")

#plt.scatter(output[:, 0], output[:,1])
#plt.scatter(output[:, 0], output[:, 2])
#plt.show()