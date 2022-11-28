#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
#Convert txt files to np arrays
directory="2mC6"
file=directory + "-500out.txt"
filename= directory + "/" + file

molecule=np.genfromtxt(filename,delimiter=",",names=True,usecols=(0,1,2,3,4))
p=molecule['pressure']
muc=molecule['muc']

plt.figure()
plt.title(file[:-7]+' molecules per unit cell')
plt.xlabel('Pressure')
plt.ylabel('muc')
plt.semilogx(p,muc,'ko',linestyle='dashed')
plt.show()