#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 
import subprocess

def seg_iast_routine(gas_frac_1, gas_frac_2, mol_1_iso, mol_2_iso, iso_name_1, iso_name_2):
#Write params to fortran-file 
    startline, stopline = 'C     Start for Python', 'C     End for Python'
    with open("../fortran/testiast.f", "r+") as file:
        data = file.readlines()

    os.remove("../fortran/testiast.f")
    for num, line in enumerate(data, 1):
        if startline in line:
            startnum = num
        elif stopline in line:
            stopnum = num
        elif 'C     Python marker 4':
            markernum = num
        elif 'end':
            endnum = num
    print(startnum, stopnum)
    del data[startnum:stopnum-1]
    #print(data)

    #for i in range(startnum + 1, startnum + 11 * no_components):
    data.insert(startnum, "      Yi(%d) = %.2fd0      \n" % (2, gas_frac_2))
    data.insert(startnum, "      Yi(%d) = %.2fd0      \n" % (1, gas_frac_1))
    data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (1, 1, mol_1_iso[0]))
    data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (1, 2, mol_1_iso[2]))
    data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (1, 1, mol_1_iso[1]))
    data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (1, 2, mol_1_iso[3]))
    data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (2, 1, mol_2_iso[0]))
    data.insert(startnum, "      Ki(%d, %d) = %.11fd0 \n" % (2, 2, mol_2_iso[2]))
    data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (2, 1, mol_2_iso[1]))
    data.insert(startnum, "      Nimax(%d, %d) = %fd0 \n" % (2, 2, mol_2_iso[3]))
    for i in range(1, 3):
        data.insert(startnum, "      Pow(%d, 1) = 1.0d0 \n" %(i))
        data.insert(startnum, "      Pow(%d, 2) = 1.0d0 \n" %(i))
        data.insert(startnum, "      Langmuir(1, 1) = .True. \n")
        data.insert(startnum, "      Langmuir(1, 2) = .True. \n")



    with open("../fortran/testiast.f", "a") as file:
        for num, line in enumerate(data, 1):
            file.write(line)


    subprocess.call(['sh', './seg_iast.sh'])

    #Move and rename current file 

    os.rename('../fortran/fort.25', "../../output/%s_%s/%s-%.2f_%s-%.2f.txt" %(iso_name_1, iso_name_2, iso_name_1, gas_frac_1, iso_name_2, gas_frac_2))
    #Return 0 for no error
    return 0;