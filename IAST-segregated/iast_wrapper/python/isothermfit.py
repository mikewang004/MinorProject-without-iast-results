#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Use this file to fit isotherms to the DS-Langmuir equation."
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 
import subprocess
import pandas as df

def return_molkg_pressure(df_iso):
    return df_iso["molkg"], df_iso["pressure"]

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def fit_DS_langmuir(df_iso, p0):
    molkg = df_iso["molkg"]
    pressure = df_iso["pressure"]
    popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = 2000)
    print(popt)
    return popt

def autofit_DS_langmuir(df_iso, p0, exp_p0, eps1, eps2):
    "Tries a curvefit until a value close enough to given p0 has been found. Returns found and valid p0."
    "First implementation only checks if all values of p0 are non-negative."
    nonneg = False
    i = 0
    ret_p0 = np.array([0,0,0,0])
    print(i)
    while i < 3:
        try:
            ret_p0 = fit_DS_langmuir(df_iso, p0)
            print("test succeded")
        except Exception as e: 
            print(e)
        i = i + 1
        #Test for non-negativity array 
        if np.invert(ret_p0.all() > 0):
            print(ret_p0 > 0)
            nonneg = True
        else:
            print("values all higher than 0")
            
    print(i)
    return ret_p0



def get_name_from_path(path):
    return path[path.rfind('/')+1:path.rfind('out')]
    


input_path_nieuw = "../../../Raspa/nieuwe_outputs/"
temp = 300
exp_p0 = [1.0e-7, 0.6, 1.0e-1, 0.7]
eps1, eps2 = 10e2, 0.1
input1 = "22mC5"
mol_1_path = input_path_nieuw + "%s/%s-%dout.txt" %(input1, input1, temp)
#input2 = "3mC6"
#mol_2_path = input_path_nieuw + "%s/%s-%dout.txt" %(input2, input2, temp)

p0 = np.array([1.0e-9, 0.7, 10e-12, 0.4])
mol_1_iso = df.read_csv(mol_1_path)
lang1 =  fit_DS_langmuir(mol_1_iso, p0)
mol1para = return_molkg_pressure(mol_1_iso)



plt.semilogx(mol1para[1], DSLangmuir(mol1para[1], lang1[0], lang1[1], lang1[2], lang1[3]), "ro", label=input1 + ", DS-Langmuir fit")
plt.semilogx(mol1para[1], mol1para[0], "go", label = input1 + ", RASPA-obtained data")
plt.title("Plot of pure-component data at %d Kelvin" %temp)
plt.legend()
plt.show()