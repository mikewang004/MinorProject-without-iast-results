#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from General_functions import ML_database
import glob
import selfies as sf

import sys

class MachineLearningInput(): 
    """INPUT: keep this order: [name molecules] , [fraction molecules], Temperature, Pressure"""
    def __init__(self, *args):
        supported_molecules = ["C7","2mC6","3mC6","22mC5","23mC5","24mC5","33mC5",  "3eC5", "223mC4"] 
        
        for item in args[0]:
            if str(item) not in supported_molecules:
                sys.exit('Input a supported molecule: "C7","2mC6","3mC6","22mC5","23mC5","24mC5","33mC5",  "3eC5", "223mC4"')
        
        try:
            self.molecules = args[0]
        except:
            sys.exit("Give input like this: MachineLearningInput([name molecules] , [fraction molecules], Temperature, Pressure)")
            
        try:
            if len(args[1]) != len(args[0]):
                sys.exit("Not for all molecules are fractions given, exit program.")
            else:
                self.fractions = args[1]
        except: 
            "Give input like this: [name molecules] , [fraction molecules], Temperature, Pressure"
            
        try:
            self.temprature = args[2]
        except:
            print("No temperature given, default initialization: T = 300K")
            self.temprature = 300
            
        try:
            self.pressure = args[3]
        except:
            print("No pressure given, default initialization: p = 1e4 Pa")
            self.pressure = 1e4
            
    def Print(self):
        """Prints properties of object"""
        print(f"Set pressure: p = {self.pressure} Pa\nSet temprature: T = {self.temprature}K")
        print("Molecules, Fraction")
        for idx in range(len(self.molecules)):
            print(str(self.molecules[idx]) + ", " + str(self.fractions[idx]))
    
    def Selfies(self):      
        """Returns a directory of all possible selfies codes"""
        
        "Creating the smiles and name arrays"
        molecule_names = ['C7',"2mC6",'3mC6','22mC5',"23mC5",'24mC5',"33mC5",  "3eC5",   "223mC4"]
        smiles_dataset = ["CCCCCCC", "CCCCC(C)C" ,"CCCC(C)CC" , "CCCC(C)(C)C" ,"CCC(C)C(C)C" , "CC(C)CC(C)C" ,"CCC(C)(C)CC","CCC(CC)CC" ,"CC(C)C(C)(C)C"]
        selfies_dataset = list(map(sf.encoder, smiles_dataset)) #transforming to selfies
        #print(selfies_dataset)
        
        max_len = max(sf.len_selfies(s) for s in selfies_dataset)
        symbols = sf.get_alphabet_from_selfies(selfies_dataset) # creating symbols for each character that is in the database
        symbols.add("[nop]") # this is an padding symbol, otherwise it does not work
        
        vocab_stoi = {symbol: idx for idx, symbol in enumerate(symbols)} #giving idx to each symbol
        
        "creating dictionary for storage"
        molecular_database = {} 
        
        for first, name in zip(selfies_dataset,molecule_names):
            'Creating one_hot encoding'
            one_hot = np.array(sf.selfies_to_encoding(first, vocab_stoi, pad_to_len =max_len)[1])
            one_hot = one_hot.reshape(one_hot.shape[1]*one_hot.shape[0])#rescaling into a vector
            
            "Adding vector to the dictionary with name"
            molecular_database[name] = one_hot
    
        return molecular_database
    
    # def 
    
    
                
    
obj = MachineLearningInput(["C7", "23mC5"], [0.7,0.3])
obj.Print()
# obj.Selfies()
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
