#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from General_functions import ML_database
import glob


class MachineLearningInput(): 
    """INPUT: keep this order: [name molecules] , [fraction molecules], Temperature, Pressure"""
    def __init__(self, *args):
        print(args[0][0])
        # for idx, item in enumerate(args):
        #         setattr(self, "attr{}".format(idx), item)
        #     else:
                
    
obj = MachineLearningInput(["a", "b"], [0.4,0.6], 300, 1e6)

#Convert txt files to np arrays
# directory="2mC6"
# file=directory + "-400out.txt"
# filename= directory + "/" + file

# molecule=np.genfromtxt(filename,delimiter=",",names=True,usecols=(0,1,2,3,4))
# p=molecule['pressure']
# muc=molecule['muc']
# print(type(molecule))

# plt.figure()
# plt.title(file[:-7]+' molecules per unit cell')
# plt.xlabel('Pressure')
# plt.ylabel('muc')
# plt.semilogx(p,muc,'ko',linestyle='dashed')
# plt.show()

# def data_gathering(path_to_output,molecular_database):
#     data={}
#     outputmaps = os.listdir(path_to_output)
#     for outputmap in outputmaps:
#         mappath = path_to_output + "/" + str(outputmap)
#         if os.path.isdir(mappath):
#             files = os.listdir(mappath)
#             for file in files:
#                 #try:
#                 paths =  mappath + "/" + str(file)
#                 label = file.split("out")[0]
                
#                 pressure,molkg,molkg_err=np.loadtxt(paths,delimiter=',',skiprows=1,usecols=(0,3,4),unpack=True)
#                 dt=np.dtype([('selfie',np.ndarray),('pressure',np.float64),('temps',int),('molkg',np.float64),('molkg_err',np.float64)])
#                 data_array= np.zeros(len(pressure), dtype=dt)
#                 temps = int(label[-3:])*np.ones(len(pressure))
                
#                 data_array['selfie']=np.repeat(molecular_database[label[:-4]][:,np.newaxis],len(pressure),axis=1).T
#                 data_array['pressure']=pressure
#                 data_array['temps']=temps
#                 data_array['molkg']=molkg
#                 data_array['molkg_err']=molkg_err
#                 print(data_array)
#                 data[label]=data_array
#                     #data[label]=np.append(molecular_database,data_array)
#     return data

# def get_data(path_to_output):
#     paths = glob.glob(path_to_output + "/*.txt")
#     for path in paths:
#         simulation = path.split("/")[-1].split("out.txt")
#         molecule = simulation[:-3]
#         temperature = simulation[-3:]
        
# path_to_out='Raspa/outputs'
# chemstructure=ML_database()
# dt=np.dtype([('selfie',np.ndarray)])
# arr=np.zeros(100,dtype=dt)
# c=np.repeat(chemstructure['C7'][:,np.newaxis],100,axis=1)
# data=data_gathering(path_to_out, chemstructure)
