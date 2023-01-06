#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Use this file to apply seg-iast."
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp
import os 
import subprocess
import pandas as df
from itertools import permutations, combinations
import shutil 

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

def load_raspa_new(temp, mol_names, mol_csvs, path = "../../../Raspa/ShrinjayOutputs/ConvertedShrinjay"):
    for root, dir, files in os.walk(path):
        for names in files:
            if str(temp) in names:
                mol_names.append(names.partition("-")[2])
                mol_csvs.append(root + "/" + names)     
    print(mol_csvs)      
    return mol_csvs, mol_names

def p0_dict(temp, path = None):
    p0_dict = {}
    if path == None:
        with open("../../new_p0/p0_values-%d.txt" %temp) as file:
            for line in file:
                name, p0 = line.split("[")
                name = name.split()[0]
                p0 = p0.replace('[', ''); p0 = p0.replace(']', '');
                p0_dict[name] = np.fromstring(p0, sep=" ")
    return p0_dict, list(p0_dict.keys())

def return_molkg_pressure(df_iso):
    return df_iso["molkg"], df_iso["pressure"]

def DSLangmuir(ab, k1, qsat1, k2, qsat2):
    k1ab, k2ab = k1 * ab, k2 * ab
    return (qsat1 * k1ab / ( 1 + k1ab)) + (qsat2 * k2ab / (1 + k2ab))

def get_mix_combinations(no_mixture, names, parallel_no, max_parallel):
    "Returns all possible combinations from a list of mixtures"
    perms = list(combinations(sorted(sorted(names), key=str.upper), no_mixture))
    del perms[::3]
    return perms[int((parallel_no-1)/max_parallel*len(perms)):int(parallel_no/max_parallel*len(perms))]



def get_frac_permutations(n_mols, n_frac = 10):
    "Returns all possible combinations of n molecules"
    gas_array = np.linspace(0, 1, n_frac+1)
    gas_fracs = np.array(list(permutations(gas_array, n_mols)))
    gas_fracs = gas_fracs[~np.any(gas_fracs == 0.0, axis=1)] #Note that an [n] mixture with one fraction = 0 is an [n] -1 mixture
    return gas_fracs[np.isclose(np.sum(gas_fracs, axis=1), 1.0), :]

def lookup_mix_dict(mix_combi, p0_dict):
    "Returns corresponding p0 values for a given value"
    p0_result = []
    for s in mix_combi:
        p0_result.append(p0_dict[s])
    return p0_result

def loop_over_list(data, start, end=None):
    "start, end both strings; data a list"
    if end == None:
        for num, line in enumerate(data, 1):
            if start in line:
                startnum = num
        return startnum
    else:
        for num, line in enumerate(data, 1):
            if start in line:
                startnum = num
            elif end in line:
                stopnum = num
        return startnum, stopnum
def prepare_strings_testiast_dotf(no_compos, mix_combi, temp):
    mix_combi = [x.strip(" ") for x in mix_combi]
    str1 = "      write(6,*) "
    str2 = "      write(6,'(2e20.10)') "
    str3 = """      write(25,'(A)') "  Pressure (Pa) @ %dK"""%(temp)
    for i in range(0, no_compos):
        if i == (no_compos -1):
            str1 += str("'Ni(%d)   '" %(i+1))
            str2 += str("Ni(%d)" %(i+1))
            #str3 += str("""     %s-%d (mol/kg)" """ %(mix_combi[i], temp))
            str3 += str(""" %s (mol/kg)" """ %(mix_combi[i]))
        else:
            str1 += str("'Ni(%d)   '," %(i+1))
            str2 += str("Ni(%d)," %(i+1))
            #str3 += str("     %s-%d (mol/kg)" %(mix_combi[i], temp))
            str3 += str(" %s (mol/kg)" %(mix_combi[i]))
    return str1, str2, str3

def prepare_write_strings_testiast_dotf(no_compos, str1):
    for i in range(0, no_compos):
        if i == (no_compos-1):
            str1 += str('Ni(%d)' %(i+1))
        else:
            str1 += str('Ni(%d),' %(i+1))
    return str1

def write_testiast_dotf(temp, mix_combi, p0_dict, gas_frac):
    "Note this is only for single iteration."
    no_compos = len(mix_combi)
    p0_array = lookup_mix_dict(mix_combi, p0_dict)
    startline, stopline = 'C     Start for Python1', 'C     End for Python1'
    start2, stop2 = 'C     Start for Python2', 'C     End for Python2'
    start3, start4 = 'C     Start for Python3', 'C     Start for Python4'
    output_strings = prepare_strings_testiast_dotf(no_compos, mix_combi, temp)
    with open("../fortran/testiast.f", "r+") as file:
        data = file.readlines()
    os.remove("../fortran/testiast.f")
    startnum, stopnum = loop_over_list(data, startline, stopline)
    del data[startnum:stopnum-1]
    startnum2, stopnum2 = loop_over_list(data, start2, stop2)
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
    startnum2, stopnum2 = loop_over_list(data, start2, stop2)
    data.insert(startnum2, output_strings[0] + "\n")
    data.insert(startnum2, output_strings[1] + "\n")
    data.insert(startnum2, output_strings[2] + "\n")
    startnum3 = loop_over_list(data, start3)
    del data[startnum3]
    data.insert(startnum3, prepare_write_strings_testiast_dotf(no_compos,"                Write(25,'(20e20.10)') P,")+ "\n")
    startnum4 = loop_over_list(data, start4)
    del data[startnum4]
    data.insert(startnum4, prepare_write_strings_testiast_dotf(no_compos,"            Write(25,'(20e20.10)') P,")+ "\n")

    with open("../fortran/testiast.f", "a") as file:
        for num, line in enumerate(data, 1):
            file.write(line)
    return 0;

def run_testiast_dotf():
    subprocess.call(['sh', './seg_iast.sh'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    #subprocess.call(['sh', './seg_iast.sh'])
    return 0;
    

def try_folder_path(temp, mix_combi, path = "../../automated_output"):
    no_compos = len(mix_combi)
    mix_combi = [x.strip(" ") for x in mix_combi]
    str1 = ""
    for i in range(0, no_compos):
        if i == no_compos-1:
            str1 += str("%s" %(mix_combi[i]))
        else:
            str1 += str("%s-" %(mix_combi[i]))
    if os.path.exists(path + "/%d_molecules/%dK_temperature/%s" % (no_compos, temp, str1)) == False:
        os.makedirs(path + "/%d_molecules/%dK_temperature/%s" % (no_compos, temp, str1))
    return path + "/%d_molecules/%dK_temperature/%s" % (no_compos, temp, str1)

def move_segiast_dotf_output(output_path, gas_frac):
    str1 = ""
    no_compos = gas_frac.shape[0]
    for i in range(0, no_compos):
        if i == no_compos-1:
            str1 += str("%.2f" %(gas_frac[i]))
        else:
            str1 += str("%.2f-" %(gas_frac[i]))
    os.rename('../fortran/fort.25', "%s/%s.txt" %(output_path, str1))

def zip_delete_dir(path):
    shutil.make_archive(path, 'zip', path)
    shutil.rmtree(path)

def seg_iast_one_combi_loop(temp, p0_lookup, mix_combi, gas_frac):
    "Computes seg-iast of all possbile permutation of a given combination of molecules."
    no_gas_iter = gas_frac.shape[0]
    for i in range(0, no_gas_iter):
        write_testiast_dotf(temp, mix_combi, p0_lookup, gas_frac[i])
        run_testiast_dotf()
        output_path = try_folder_path(temp, mix_combi)
        move_segiast_dotf_output(output_path, gas_frac[i])
    return 0;

def automatic_seg_iast(temp, p0_lookup, mix_combi, gas_frac):
    "Computes seg-iast all possbile combinations of [n] molecules."
    no_mol_iter = np.shape(mix_combi)
    for i in range(0, no_mol_iter[0]):
        seg_iast_one_combi_loop(temp, p0_lookup, mix_combi[i], gas_frac)
        zip_delete_dir(try_folder_path(temp, mix_combi[i]))
    return 0;

def automatic_temp_seg_iast(temp_list, p0_lookup, mix_combi, gas_frac):
    "Computes the seg-iast of all temperatures of all possible combinations of [n] moletules"
    for temp in temp_list:
        automatic_seg_iast(temp, p0_lookup, mix_combi, gas_frac)
    return 0;

def automatic_no_mols_seg_iast(temp, p0_lookup,names, mols, mix_combi, max_no_mols = 5, low_no_frac = 10, high_no_frac = 20):
    for no_molecules in mols:
        if no_molecules > 3:
            no_gas_fractions = high_no_frac
        else:
            no_gas_fractions = low_no_frac
        gas_frac = get_frac_permutations(int(no_molecules), int(no_gas_fractions))
        automatic_seg_iast(temp, p0_lookup, mix_combi, gas_frac)
def main():
    temp = 550
    p0_lookup, names = p0_dict(temp)
    start_mol = 2
    no_molecules = 5
    mix_combi = get_mix_combinations(no_molecules, names, 14, 16)
    mols = np.arange(start_mol, no_molecules + 1) 
    automatic_no_mols_seg_iast(temp, p0_lookup, names, mols, mix_combi)


if __name__ == "__main__":
    main()  