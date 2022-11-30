#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 
import subprocess

startnum, stopnum = 0,0
input_path = "../../Raspa/outputs/"
input_path_nieuw = "../../Raspa/nieuwe_outputs/"
#mol_1_path = input_path + "C7/C7-500out.txt"
#mol_2_path = input_path + "3mC6/3mC6-500out.txt"
mol_1_path = input_path_nieuw + "2mC6/2mC6-500out.txt"
mol_2_path = input_path_nieuw + "3mC6/3mC6-500out.txt"
molecule_1 = r"$2mC6$"
molecule_2 = r"$3mC6$"
temp = r"$500 K$"
no_components = 2 

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def fit_DS_langmuir(path, p0):
    df_iso = pd.read_csv(path)
    molkg = df_iso["molkg"]
    pressure = df_iso["pressure"]
    popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0)
    print(popt)
    return popt

def return_molkg_pressure(df_iso):
    q_tot = df_iso["molkg"]
    ab = df_iso["pressure"]
    return q_tot, ab

def seg_iast_routine(gas_frac_1, gas_frac_2, mol_1_iso, mol_2_iso):
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
        elif 'C     Python marker 4':
            markernum = num
        elif 'end':
            endnum = num
    print(startnum, stopnum)
    del data[startnum:stopnum-1]
    #print(data)

    #for i in range(startnum + 1, startnum + 11 * no_components):
    data.insert(startnum, "      Yi(%d) = %.2fd0      \n" % (2, gas_frac_2))
    data.insert(startnum, "      Yi(%d) = %.2fd0      \n" % (1, gas_frac_1))
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



    with open("fortran/testiast.f", "a") as file:
        for num, line in enumerate(data, 1):
            file.write(line)


    subprocess.call(['sh', './seg_iast.sh'])

    #Move and rename current file 

    os.rename('fortran/fort.25', "../output/22mC5-500_23mC5-500/22mC5-500-%.2f_23mC5-500-%.2f.txt" %(gas_frac_1, gas_frac_2))
    #Return 0 for no error
    return 0;


#mol_1_iso = fit_DS_langmuir(mol_1_path, [1.0e-5, 0.701, 1.0e-4, 1.0]) #C7-500
#mol_2_iso = fit_DS_langmuir(mol_2_path, [1.0e-11, 0.6, 1.0e-6, 0.8]) #3mC6-500
mol1para = return_molkg_pressure(pd.read_csv(mol_1_path)) 
mol2para = return_molkg_pressure(pd.read_csv(mol_2_path)) 


mol_1_iso = fit_DS_langmuir(mol_1_path, [1.0e-4, 0.2, 1.0e-4, 0.6]) #22mC5
mol_2_iso = fit_DS_langmuir(mol_2_path, [1.0e-2, 0.5, 1.0e-4, 0.2]) #23mC5




plt.semilogx(mol1para[1], DSLangmuir(mol1para[1], mol_1_iso[0], mol_1_iso[1], mol_1_iso[2], mol_1_iso[3]), "ro", label=molecule_1 + r", homogeneous gas")
plt.semilogx(mol1para[1], mol1para[0], "go")
#plt.semilogx(mol2para[1], mol2para[0], "bo")
#plt.semilogx(mol2para[1], DSLangmuir(mol2para[1], mol_2_iso[0], mol_2_iso[1], mol_2_iso[2], mol_2_iso[3]), "r.", label=molecule_2 + r", homogeneous gas")
#plt.title("Loadings of a " + molecule_1 + " and " + molecule_2 +  " mixture at temperature " + temp)
#plt.xlabel("Pressure (Pascal)")
#plt.ylabel("Loading (mol/kg)")
#plt.legend()
plt.show()

maxnofrac = 50
gasfrac = np.array([np.linspace(0, 1, maxnofrac), np.linspace(1, 0, maxnofrac)])
#for i in range(0, maxnofrac):
    #seg_iast_routine(gasfrac[0][i], gasfrac[1][i], mol_1_iso, mol_2_iso)