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
    print(data)


data = np.loadtxt("text.txt")
print(data[:, 80])
LangmuirParaSelection(data)