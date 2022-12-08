#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Use this file to apply seg-iast."
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast
import scipy as sp
import os 
import subprocess
import pandas as df
from itertools import combinations

temp = 400
def load_raspa(temp, path = "../../../Raspa/nieuwe_outputs"):
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

def read_p0(path = None):
    if path == None:
        with open("p0_values-%d.txt" %temp) as file:
            data = [[(x) for x in line.split()] for line in file]
        print(data[2])
    return data

def return_molkg_pressure(df_iso):
    return df_iso["molkg"], df_iso["pressure"]

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def get_combinations(no_mixture, names):
    "Returns all possible combinations from a list of mixtures"
    names_int = np.ones(len(names))
    for i in range(0, len(names)):
        names_int[i] = names_int[i] * (1 + i)
    print(names_int)
    perms = list(combinations(names_int, no_mixture))
    return perms

def gas_frac_n(n_mols, n_frac = 10):
    "Returns all possible combinations of n molecules"
    #gas_array = np.ones(n_frac)
    #for i in range(0, n_frac):
        #gas_array[i] = gas_array[i] * (i + 1)/10
    gas_array = np.linspace(0, 1, n_frac)
    gas_fracs = np.array(list(combinations(gas_array, n_mols)))
    print(gas_fracs[np.isclose(np.sum(gas_fracs, axis=1), 1.0), :])
    return gas_fracs[np.isclose(np.sum(gas_fracs, axis=1), 1.0), :]




def main():
    _, names = load_raspa(temp)
    data1 = get_combinations(2, names)
    gas_frac_n(2)

if __name__ == "__main__":
    main()  