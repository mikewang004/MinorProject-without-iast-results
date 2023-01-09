#!/usr/bin/env python3
# -*- coding: utf-8 -*
import os 
import shutil

def load_raspa(temp, path = "."):
    "Can also be applied to search for general strings I think"
    mol_names = []
    mol_csvs = []
    for root, dir, files in os.walk(path):
        for names in files:
            if str(temp) in names:
                mol_names.append(names.partition("-")[0])
                #print(names)
                mol_csvs.append(root + "/" + names)             
    return mol_csvs, mol_names

def check_if_iast_wrapper_exists(total_no_parallelisation, path = ".", delftblue_jobmax = 16):
    for i in range(0, delftblue_jobmax):
        if i+1 <= total_no_parallelisation:
            if os.path.exists("iast_wrapper-%d" %(i+1)) == False:
                shutil.copytree("iast_wrapper_template", "iast_wrapper-%d" %(i+1))
        else:
            if os.path.exists("iast_wrapper-%d" %(i+1)) == True:
                shutil.rmtree("iast_wrapper-%d" %(i+1))

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

def modify_autosegiast(path, temp, startno_molecules, no_molecules, itr, total_no_parallelisation):
    with open("autosegiast-template.py", "r+") as file:
        data = file.readlines()
    startnum = loop_over_list(data, "            no_gas_fractions = low_no_frac")
    del data[startnum]
    data.insert(startnum, "        mix_combi = get_mix_combinations(no_molecules, names, %d, %d)\n" %(itr, total_no_parallelisation))
    startnum = loop_over_list(data, "def main():")
    for i in range(0, 3):
        del data[startnum]
    data.insert(startnum, "    no_molecules = %d\n" %(no_molecules)) #does not work with \t instead of 4 spaces
    data.insert(startnum, "    start_mol = %d\n" %(startno_molecules))
    data.insert(startnum, "    temp = %d\n" %(temp))
    os.remove(path)
    with open(path, "w") as file:
        for num, line in enumerate(data, 1):
            file.write(line)
    return 0;

def loop_modify_autosegiast(path, temp, startno_molecules, no_molecules, total_no_parallelisation):
    for i in range(0, total_no_parallelisation):
        modify_autosegiast(path[i], temp, startno_molecules, no_molecules, i+1, total_no_parallelisation)

def write_wrapper(temp, start_mol, no_molecules, total_no_parallelisation):
    rename_template_seg_iast(True)
    check_if_iast_wrapper_exists(total_no_parallelisation)
    path_list, _ = load_raspa("autosegiast.py")
    path_list = list(sorted(sorted(path_list), key=str.upper))[:-1]
    loop_modify_autosegiast(path_list, temp, start_mol, no_molecules, total_no_parallelisation)
    rename_template_seg_iast(False)

def rename_template_seg_iast(to_txt):
    "to txt == True -> converts from .sh to .txt; to txt == False -> from .txt to .sh"
    str1 = "iast_wrapper_template/python/segiast_delftblue.sh"
    str2 = "iast_wrapper_template/python/segiast_delftblue.txt"
    if to_txt == False:
        os.rename(str1, str2)
    else:
        os.rename(str2, str1)
def main():
    temp = 550 
    start_mol = 5 #lower bound of how many molecules there shoudl be in a mixture
    no_molecules = 5 #upper bound of how many molecules should be in a mixture
    total_no_parallelisation = 10 #total number of parallel processes
    write_wrapper(temp, start_mol, no_molecules, total_no_parallelisation)

if __name__ == "__main__":
    main()  
