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

def LangmuirParaSelection(data):
    "Deletes all nan and all rows for which a negative value has been found."
    "Note structure is [4, m]."
    data = np.delete(data, np.isnan(data).any(axis=0), axis=1)
    data = data[:, data.min(axis=0)>=0.0]
    return data

input_path = "../../../Raspa/outputs/"
input_path_nieuw = "../../../Raspa/nieuwe_outputs/"
temp = 400
input1 = "23mC5"
#mol_1_path = input_path_nieuw + "%s/%s-%dout.txt" %(input1, input1, temp)
mol_1_path = input_path + "%s/%s-%dout.txt" %(input1, input1, temp)

data = np.loadtxt("p0_output/%s-%dp0.txt" % (input1, temp))
mol1para = return_molkg_pressure(df.read_csv(mol_1_path))

sel_data = LangmuirParaSelection(data)
print(sel_data)
for i in range(0, sel_data.shape[1]):
    plt.loglog(mol1para[1], DSLangmuir(mol1para[1], sel_data[0,i], sel_data[1,i], sel_data[2,i], sel_data[3,i]), "r.", label=input1 + ", DS-Langmuir fit")
    plt.loglog(mol1para[1], mol1para[0], "go", label = input1 + ", RASPA-obtained data")

plt.show()