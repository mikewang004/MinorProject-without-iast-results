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
from itertools import permutations, combinations

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
    return mol_csvs, mol_names

def p0_dict(temp, path = None):
    p0_dict = {}
    if path == None:
        with open("p0_values-%d.txt" %temp) as file:
            for line in file:
                name, p0 = line.split("\t")
                p0 = p0.replace('[', ''); p0 = p0.replace(']', '');
                p0_dict[name] = np.fromstring(p0, sep=" ")
    return p0_dict, list(p0_dict.keys())

def return_molkg_pressure(df_iso):
    return df_iso["molkg"], df_iso["pressure"]

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def get_mix_combinations(no_mixture, names):
    "Returns all possible combinations from a list of mixtures"
    perms = list(combinations(names, no_mixture))
    return perms

def get_frac_permutations(n_mols, n_frac = 10):
    "Returns all possible combinations of n molecules"
    gas_array = np.linspace(0, 1, n_frac+1)
    gas_fracs = np.array(list(permutations(gas_array, n_mols)))
    return gas_fracs[np.isclose(np.sum(gas_fracs, axis=1), 1.0), :]

def lookup_mix_dict(mix_combi, p0_dict):
    "Returns corresponding p0 values for a given value"
    p0_result = []
    for s in mix_combi:
        p0_result.append(p0_dict[s])
    return p0_result

def prepare_strings_testiast_dotf(no_compos, mix_combi, temp):
    str1 = "      write(6,*) "
    str2 = "      write(6,'(2e20.10)') "
    str3 = "      write(25,'(A)')   '     Pressure (Pa)"
    for i in range(0, no_compos):
        if i == (no_compos -1):
            str1 += str('Ni(%d)   ' %(i+1))
            str2 += str("Ni(%d)" %(i+1))
            str3 += str("     %s-%d (mol/kg)'" %(mix_combi[i], temp))
        else:
            str1 += str("'Ni(%d)   '," %(i+1))
            str2 += str("Ni(%d)," %(i+1))
            str3 += str("     %s-%d (mol/kg)" %(mix_combi[i], temp))
    return str1, str2, str3

def write_testiast_dotf(temp, mix_combi, p0_dict, gas_frac):
    "Note this is only for single iteration."
    no_compos = len(mix_combi)
    p0_array = lookup_mix_dict(mix_combi, p0_dict)
    startline, stopline = 'C     Start for Python1', 'C     End for Python1'
    start2, stop2 = 'C     Start for Python2', 'C     End for Python2'
    output_strings = prepare_strings_testiast_dotf(no_compos, mix_combi, temp)
    with open("debug/testiast.f", "r+") as file:
        data = file.readlines()
    os.remove("debug/testiast.f")
    for num, line in enumerate(data, 1):
        if startline in line:
            startnum = num
        elif stopline in line:
            stopnum = num
        elif start2 in line:
            startnum2 = num
        elif stop2 in line:
            stopnum2 = num
    del data[startnum:stopnum-1]

    del data[startnum2:stopnum2-1]
    for i in range(0, no_compos):
        j = i + 1
        data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (j, 1, p0_array[i][0]))
        data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (j, 2, p0_array[i][2]))
        data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (j, 1, p0_array[i][1]))
        data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (j, 2, p0_array[i][3]))
        data.insert(startnum, "      Pow(%d, 1) = 1.0d0 \n" %(j))
        data.insert(startnum, "      Pow(%d, 2) = 1.0d0 \n" %(j))
        data.insert(startnum, "      Langmuir(%d, 1) = .True. \n" %(j))
        data.insert(startnum, "      Langmuir(%d, 2) = .True. \n" %(j))
        data.insert(startnum, "      Yi(%d) =  %.2fd0 \n" %(j, gas_frac[i]))
    data.insert(startnum, "      Ncomp = %d \n" %(no_compos))
    data.insert(startnum2, output_strings[0] + "\n")
    data.insert(startnum2, output_strings[1] + "\n")
    data.insert(startnum2, output_strings[2] + "\n")

    with open("debug/testiast.f", "a") as file:
        for num, line in enumerate(data, 1):
            file.write(line)
    return 0;

def run_testiast_dotf(temp, i):
    subprocess.call(['sh', './seg_iast.sh'])
    os.rename('../fortran/fort.25', "debugoutput-%d-%i.txt" %(temp, i))
    return 0;
    #Return 0 for no error
def main():
    p0_lookup, names = p0_dict(400)
    print(names)
    mix_combi = get_mix_combinations(3, names)
    gas_frac = get_frac_permutations(3, 10)
    write_testiast_dotf(temp, mix_combi[3], p0_lookup, gas_frac[0])
if __name__ == "__main__":
    main()  