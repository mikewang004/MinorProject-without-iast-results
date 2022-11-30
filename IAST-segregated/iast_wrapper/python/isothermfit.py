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


def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def fit_DS_langmuir(path, p0):
    df_iso = pd.read_csv(path)
    molkg = df_iso["molkg"]
    pressure = df_iso["pressure"]
    popt, pcov = sp.optimize.curve_fit(DSLangmuir, pressure, molkg, p0=p0, maxfev = 10000)
    print(popt)
    return popt

def autofit_DS_langmuir(path, p0, exp_p0, eps1, eps2):
    ret_p0 = np.array([0,0,0,0])
    with np.all(ret_p0)
        ret_p0 = fit_DS_langmuir(path, p0)
        p0[0] = p0[0]*0.1

def get_name_from_path(path):
    return path[path.rfind('/')+1:path.rfind('out')]
    
input_path_nieuw = "../../../Raspa/nieuwe_outputs/"
temp, p0 = 500, [1.0e-7, 0.6, 1.0e-4, 0.7]
exp_p0 = [1.0e-7, 0.6, 1.0e-1, 0.7]
eps1, eps2 = 10e2, 0.1
input1 = "22mC5"
mol_1_path = input_path_nieuw + "%s/%s-%dout.txt" %(input1, input1, temp)
#input2 = "3mC6"
#mol_2_path = input_path_nieuw + "%s/%s-%dout.txt" %(input2, input2, temp)


iso_1=  autofit_DS_langmuir(mol_1_path, p0, exp_p0, eps1, eps2)