#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Use this file to fit isotherms to the DS-Langmuir equation."
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp
import os 
import subprocess
import pandas as df
import scipy.stats

plotpath = "../../langmuir_pure_plots/manual_plots/"

def return_molkg_pressure(df_iso):
    try:
        molkg, pressure = df_iso["molkg"], df_iso["pressure"]
    except:
        molkg, pressure = df_iso["molkg"], df_iso["# pressure"]
    return molkg, pressure

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def fit_DS_langmuir(df_iso, p0, maxfev = 2000):
    molkg, pressure = return_molkg_pressure(df_iso)
    popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = maxfev)
    print(popt)
    return popt

def get_isotherm(temp, name, shrinjay=True):
    if shrinjay == True:
        input_loc = "../../../Raspa/ShrinjayOutputs/ConvertedShrinjay/%dK-%s" %(temp, name)
    else:
        input_loc = "../../../Raspa/outputs/%s/%s-%dout.txt" %(name, name, temp)
    return input_loc

def save_p0_plot(mol1iso, temp, sel_data, input1,plotpath = plotpath):
    plt.figure()
    mol1para = return_molkg_pressure(mol1iso)
    plt.loglog(mol1para[1], DSLangmuir(mol1para[1], sel_data[0], sel_data[1], sel_data[2], sel_data[3]), "r.", label=input1 + ", DS-Langmuir fit w/ p0 = %s" %(sel_data))
    plt.loglog(mol1para[1], mol1para[0], "go", label = input1 + ", RASPA-obtained data")
    plt.legend(prop={'size':6})
    plt.title("Dual-Side Langmuir fit of %s" %input1)
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("Loading (mol/kg)")
    plt.savefig(plotpath + "manual_%s-%d_plot" %(input1, temp))
    plt.show()
    plt.close()

def fit_manual(temp, name,p0, shrinjay=True, maxfev = 2000):
    # Read file first 
    iso = df.read_csv(get_isotherm(temp, name, shrinjay))
    popt = fit_DS_langmuir(iso, p0, maxfev = maxfev)
    save_p0_plot(iso, temp, popt, name)
    return 0;

def main():
    temp = 600
    name = "33mC5"
    p0 = np.array([10e-6, 0.7, 10e-13, 10e0])
    fit_manual(temp, name, p0, False, 10000)

if __name__ == "__main__":
    main()  