#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 
import subprocess
def return_molkg_pressure(df_iso):
    return df_iso["molkg"], df_iso["pressure"]

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def fit_DS_langmuir(path, p0):
    df_iso = pd.read_csv(path)
    molkg = df_iso["molkg"]
    pressure = df_iso["pressure"]
    popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = 2000)
    print(popt)
    return popt

def seg_iast_routine(gas_frac_1, gas_frac_2, mol_1_iso, mol_2_iso):
#Write params to fortran-file 
    startline, stopline = 'C     Start for Python', 'C     End for Python'
    with open("../fortran/testiast.f", "r+") as file:
        data = file.readlines()

    os.remove("../fortran/testiast.f")
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



    with open("../fortran/testiast.f", "a") as file:
        for num, line in enumerate(data, 1):
            file.write(line)


    subprocess.call(['sh', './seg_iast.sh'])

    #Move and rename current file 
    
    try: 
        os.rename('../fortran/fort.25', "../../output/%s-%d_%s-%d/%s-%d-%.2f_%s-%d-%.2f.txt" %(input1, temp, input2, temp, \
        input1, temp, gas_frac_1, input2, temp, gas_frac_2))
    except:
        os.makedirs("../../output/%s-%d_%s-%d" % (input1, temp, input2, temp))
        os.rename('../fortran/fort.25', "../../output/%s-%d_%s-%d/%s-%d-%.2f_%s-%d-%.2f.txt" %(input1, temp, input2, temp, \
        input1, temp, gas_frac_1, input2, temp, gas_frac_2))
    #Return 0 for no error
    return 0;

maxnofrac = 50
gasfrac = np.array([np.linspace(0, 1, maxnofrac), np.linspace(1, 0, maxnofrac)])

input1 = "2mC6"
input2 = "3mC6"
temp = 400
input_path = "../../../Raspa/outputs/"
input_path_nieuw = "../../../Raspa/nieuwe_outputs/"
mol_1_path = input_path + "%s/%s-%dout.txt" %(input1, input1, temp)
mol_2_path = input_path + "%s/%s-%dout.txt" %(input2, input2, temp)

mol_1_iso = fit_DS_langmuir(mol_1_path, [2.519e-03, 6.751e-01, 3.741e-09, 8.313e-01]) 
mol_2_iso = fit_DS_langmuir(mol_2_path, [1.567e-03, 6.801e-01, 3.045e-09, 8.156e-01]) 

for i in range(0, maxnofrac):
    seg_iast_routine(gasfrac[0][i], gasfrac[1][i], mol_1_iso, mol_2_iso)

