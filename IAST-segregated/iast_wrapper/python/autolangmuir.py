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


plotpath = "../../langmuir_pure_plots"
temp = 300
def load_raspa(temp, path = "../../../Raspa"):
    mol_names = []
    mol_csvs = []
    for root, dir, files in os.walk(path):
        for names in files:
            if str(temp) in names:
                mol_names.append(names.partition("-")[0])
                #print(names)
                mol_csvs.append(root + "/" + names)
    print(mol_csvs)                
    return mol_csvs, mol_names


def return_molkg_pressure(df_iso):
    return df_iso["molkg"], df_iso["pressure"]

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def try_curvefit(DSLangmuir, pressure, molkg, p0, maxfev = 2000):
    try: 
        popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = maxfev)
        p0 = popt
    except Exception as e: 
        print(e) 
        p0 = np.array([np.nan, np.nan, np.nan, np.nan])
    return p0

def iterative_DS_Langmuir(df_iso, k1_its=6, q_its=4, q1_value=0.7): 
    "Generates various p0 starting values, then generates curvefits for all of them"
    qlinspace = np.linspace(0.6, 0.8, q_its)
    klogspace = np.logspace(0, -12, k1_its)
    molkg, pressure = df_iso["molkg"], df_iso["pressure"]
    p0_array = np.zeros([4, k1_its * k1_its * q_its * q_its])
    m = 0 #helper variable
    print("Started for-loop.")
    for i in range(0, k1_its):#loop first over k1,q1 
        for k in range(0, k1_its):
            for l in range (0, q_its):
                p0 = np.array([klogspace[i], q1_value, klogspace[k], qlinspace[l]])
                p0_array[:, m] = try_curvefit(DSLangmuir, pressure, molkg, p0=p0, maxfev = 2000)
                m = m + 1
    return(p0_array)

def autoselect_p0_DS_Langmuir(data):
    "Selects data based upon two criteria: 1. no values should be smaller than 0;"
    "2. q1 should be between 0.6 and 0.8."
    "Returns the average p0 value."
    data = data[:, data.min(axis=0)>=10e-21]
    data = data[:, np.logical_and(data[1, :]>0.6, data[1,:]< 0.8)]
    return np.average(data, axis=1)

def save_p0_plot(mol1iso, sel_data, input1):
    plt.figure()
    mol1para = return_molkg_pressure(mol1iso)
    plt.loglog(mol1para[1], DSLangmuir(mol1para[1], sel_data[0], sel_data[1], sel_data[2], sel_data[3]), "r.", label=input1 + ", DS-Langmuir fit")
    plt.loglog(mol1para[1], mol1para[0], "go", label = input1 + ", RASPA-obtained data")
    plt.legend()
    plt.title("Dual-Side Langmuir fit of %s." %input1)
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("Loading (mol/kg)")
    plt.savefig(plotpath + "%s-%d_plot" %(input1, temp))
    plt.clear()

def auto_fit_plot_Langmuir(temp):
    paths, mol_names = load_raspa(temp)
    #print(paths)
    print(mol_names)

    for i in range(0, len(mol_names)):
        mol_1_iso, name = df.read_csv(paths[i]), mol_names[i]
        mol1para = return_molkg_pressure(mol_1_iso)
        sel_data = autoselect_p0_DS_Langmuir(iterative_DS_Langmuir(mol_1_iso))
        save_p0_plot(mol_1_iso, sel_data, name)
        print("Finised molecule %s." %name)

    return 0;

auto_fit_plot_Langmuir(temp)