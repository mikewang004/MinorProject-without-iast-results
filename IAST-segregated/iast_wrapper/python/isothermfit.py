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
    "Note per 01/12/22: probably stupid approach."
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

def try_curvefit(DSLangmuir, pressure, molkg, p0, maxfev = 2000):
    try: 
        popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = maxfev)
        p0 = popt
    except Exception as e: 
        print(e) 
        p0 = np.array([np.nan, np.nan, np.nan, np.nan])
    return p0
def iterative_DS_Langmuir(df_iso, k1_its, q_its, Double_Side=True): 
    "Generates various p0 starting values, then generates curvefits for all of them"
    #p0 = [10e-1, 0.6, 10e-1, 0.6]
    qlinspace = np.linspace(0.5, 4, q_its)
    klogspace = np.logspace(0, -12, k1_its)
    molkg, pressure = df_iso["molkg"], df_iso["pressure"]
    p0_array = np.zeros([4, k1_its * k1_its * q_its * q_its])
    m = 0 #helper variable
    if Double_Side == True:
        k2logspace = klogspace
        q2linspace = qlinspace
        print("true")
    else:
        k2logspace = np.zeros([k1_its])
        q2linspace = np.zeros([q_its])
        print("false")
    qlinspace = np.ones([q_its]) * 0.7
    print("Started for-loop.")
    for i in range(0, k1_its):#loop first over k1,q1 
        if Double_Side == True:
            #print("double_side still true")
            for k in range(0, k1_its):
                for l in range (0, q_its):
                    p0 = np.array([klogspace[i], 0.7, k2logspace[k], q2linspace[l]])
                    p0_array[:, m] = try_curvefit(DSLangmuir, pressure, molkg, p0=p0, maxfev = 2000)
                    m = m + 1
                    #print("Finished k = %d, l = %d iteration." %(k, l))
        else:
            p0 = np.array([klogspace[i], qlinspace[j], k2logspace[i], q2linspace[j]])
            p0_array[:, m] = try_curvefit(DSLangmuir, pressure, molkg, p0=p0, maxfev = 2000)
            m = m + 1
        #print("Finished i = %d, j = %d iteration." %(i, j))
    return(p0_array)

def get_name_from_path(path):
    return path[path.rfind('/')+1:path.rfind('out')]
    

input_path = "../../../Raspa/outputs/"
input_path_nieuw = "../../../Raspa/nieuwe_outputs/"
temp = 500
exp_p0 = [1.0e-7, 0.6, 1.0e-1, 0.7]
eps1, eps2 = 10e2, 0.1
input1 = "24mC5"
mol_1_path = input_path_nieuw + "%s/%s-%dout.txt" %(input1, input1, temp)
#input2 = "3mC6"
#mol_2_path = input_path_nieuw + "%s/%s-%dout.txt" %(input2, input2, temp)

p0 = np.array([1.0e-9, 0.7, 10e-12, 0.4])
mol_1_iso = df.read_csv(mol_1_path)
lang1 =  fit_DS_langmuir(mol_1_iso, p0)
mol1para = return_molkg_pressure(mol_1_iso)

data = iterative_DS_Langmuir(mol_1_iso, 6, 6, Double_Side=True)
np.savetxt("p0_output/%s-%dp0.txt" % (input1, temp), data)


#plt.semilogx(mol1para[1], DSLangmuir(mol1para[1], lang1[0], lang1[1], lang1[2], lang1[3]), "ro", label=input1 + ", DS-Langmuir fit")
#plt.semilogx(mol1para[1], mol1para[0], "go", label = input1 + ", RASPA-obtained data")
#plt.title("Plot of pure-component data at %d Kelvin" %temp)
#plt.legend()
#plt.show()