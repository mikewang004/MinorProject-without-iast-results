#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 

startnum, stopnum = 0,0
input_path = "../../Raspa/outputs/"

def return_langmuir(df_iso):
    return pyiast.ModelIsotherm(df_iso, loading_key = "molkg", pressure_key = "pressure", model = "DSLangmuir")


def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def fit_DS_langmuir(df_iso, p0):
    molkg = df_iso["molkg"]
    pressure = df_iso["pressure"]
    popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0)
    #print(q_tot, ab)
    print(popt)
    return popt

def return_molkg_pressure(df_iso):
    q_tot = df_iso["molkg"]
    ab = df_iso["pressure"]
    return q_tot, ab

molecule_1 = r"$CH_7$"
molecule_2 = r"$3mC6$"
temp = r"$500 K$"
no_components = 2 
#moleculeiso_1 = return_langmuir(pd.read_csv(input_path + "C7/C7-500out.txt"))
#moleculeiso_2 = return_langmuir(pd.read_csv(input_path + "3mC6/3mC6-500out.txt"))

#moleculeiso_1.print_params()
#moleculeiso_2.print_params()

mol_1_iso = fit_DS_langmuir(pd.read_csv(input_path + "C7/C7-500out.txt"), [1.0e-5, 0.701, 1.0e-4, 1.0])
mol_2_iso = fit_DS_langmuir(pd.read_csv(input_path + "3mC6/3mC6-500out.txt"), [1.0e-11, 0.6, 1.0e-6, 0.8])

no_fracs = 2
no_pressures = 19
gas_frac = np.linspace(0.5, 0.5, no_fracs)
partial_pressures = np.logspace(0, 9, num=no_pressures)
partial_pressures2 = np.logspace(0, 9, num=25)
mol1para = return_molkg_pressure(pd.read_csv(input_path + "C7/C7-500out.txt"))
mol2para = return_molkg_pressure(pd.read_csv(input_path + "3mC6/3mC6-500out.txt"))
#plt.semilogx(partial_pressures, DSLangmuir(mol1para[1], mol_1_iso[0], mol_1_iso[1], mol_1_iso[2], mol_1_iso[3]), "ro", label=molecule_1 + r", homogeneous gas")
#plt.semilogx(mol1para[1], mol1para[0], "go")
plt.semilogx(mol2para[1], mol2para[0], "bo")
plt.semilogx(partial_pressures2, DSLangmuir(mol2para[1], mol_2_iso[0], mol_2_iso[1], mol_2_iso[2], mol_2_iso[3]), "r.", label=molecule_2 + r", homogeneous gas")
plt.title("Loadings of a " + molecule_1 + " and " + molecule_2 +  " mixture at temperature " + temp)
plt.xlabel("Pressure (Pascal)")
plt.ylabel("Loading (mol/kg)")
plt.legend()
plt.show()

#Write params to fortran-file 
startline, stopline = 'C     Start for Python', 'C     End for Python'
with open("fortran/testiast.f", "r+") as file:
    data = file.readlines()

os.remove("fortran/testiast.f")
for num, line in enumerate(data, 1):
    if startline in line:
        startnum = num
    elif stopline in line:
        stopnum = num

print(startnum, stopnum)
del data[startnum:stopnum-1]
#print(data)

#for i in range(startnum + 1, startnum + 11 * no_components):

data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (1, 1, mol_1_iso[0]))
data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (1, 2, mol_1_iso[2]))
data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (1, 1, mol_1_iso[1]))
data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (1, 2, mol_1_iso[3]))
data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (2, 1, mol_2_iso[0]))
data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (2, 2, mol_2_iso[2]))
data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (2, 1, mol_2_iso[1]))
data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (2, 2, mol_2_iso[3]))
for i in range(1, 3):
    data.insert(startnum, "      Pow(%d, 1) = 1.0d0 \n" %(i))
    data.insert(startnum, "      Pow(%d, 2) = 1.0d0 \n" %(i))
    data.insert(startnum, "      Langmuir(1, 1) = .True. \n")
    data.insert(startnum, "      Langmuir(1, 2) = .True. \n")


#print(data)

with open("fortran/testiast.f", "a") as file:
    for num, line in enumerate(data, 1):
        file.write(line)

#Read and plot output fortran-file 

#output = np.loadtxt("fortran/fort.25")

#plt.scatter(output[:, 0], output[:,1])
#plt.scatter(output[:, 0], output[:, 2])
#plt.show()