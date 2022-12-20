"""
Created on Mon Dec 12 11:15:10 2022

@author: Steven
"""
import matplotlib.pyplot as plt
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
print(tf.__version__)
import selfies as sf
import numpy as np
import os
import pandas as pd
import glob
import sys

def ML_database():

    "Creating the smiles and name arrays"
    molecule_names = ['C7',"2mC6",'3mC6','22mC5',"23mC5",'24mC5',"33mC5",  "3eC5",   "223mC4"]
    smiles_dataset = ["CCCCCCC", "CCCCC(C)C" ,"CCCC(C)CC" , "CCCC(C)(C)C" ,"CCC(C)C(C)C" , "CC(C)CC(C)C" ,"CCC(C)(C)CC","CCC(CC)CC" ,"CC(C)C(C)(C)C"]
    selfies_dataset = list(map(sf.encoder, smiles_dataset)) #transforming to selfies
    
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
def make_training_database_ver2(max_amount_mols, chemstructure=ML_database()):
    path_RASPA=glob.glob('MachineLearning/Outputs_RASPA/*.txt')
    path_IAST=glob.glob('IAST-segregated/automated_output/*')
    
    data_RASPA=[]
    data_IAST=[]
    # print(path_IAST)
    # sys.exit(0)
    
    for file in path_RASPA:
        file = file.replace("\\", "/") #comment this line if you use linux
        molecule = file.split('/')[-1].split('-')[0]

        data = np.loadtxt(file,skiprows=1,delimiter=',',usecols=(0,1,-1))  
        selfie=np.repeat(chemstructure[molecule], data.shape[0]).reshape(52,data.shape[0]).T
        data=np.hstack((selfie,data))
        data_RASPA.append(data)
    
    for path_amount in path_IAST:
        path_amount = path_amount.replace("\\", "/")
        # print(path_amount)
        amount_mols = int(path_amount.split('/')[-1].split('_')[0])
        # print(amount_mols)
        if amount_mols>max_amount_mols:
            break
        path_temps = glob.glob(path_amount + "/*")
        for folder in path_temps:
            folder = folder.replace("\\", "/") #comment this line if you use linux
            # print(folder)
            # print(folder.split('/')[-1].split("K")[0])
            temp = float(folder.split('/')[-1].split("K")[0])
            # print(folder + "/*.txt")
            molc_folder = glob.glob(folder + "/*")
            
            # print(molc_folder)
            for mols_mix in molc_folder:
                mols_mix = mols_mix.replace("\\", "/")
                # print(mols_mix)
                # print(type(amount_mols))
                mols = mols_mix.split('/')[-1].split('-')
                # print(molc_folder)
                path_fracs = glob.glob(mols_mix + "/*.txt")
                for file in path_fracs:
                    file = file.replace("\\", "/")
                    fracs = file.split('/')[-1].split('.txt')[0].split("-")
                    # print(fracs)
                    fracs = np.array(fracs, dtype=float)
                    # print(path_fracs)
                    # print(fracs)
                    try:
                        data=np.loadtxt(file,delimiter='   ',skiprows =1, usecols=(range(1,amount_mols+2)))#Pressure, loading m1, loading m2,..., loading mi
                    except:
                        print(f"failed to load file: {file}")
                        continue
                    if len(mols)<max_amount_mols:
                        # print(data.shape)
                        # print(np.zeros((max_amount_mols - len(mols), len(data))).shape)
                        data = np.append(data, np.zeros((len(data),max_amount_mols - len(mols))), axis =1)
                    
                    
                    # print(data.shape)
                    selfie = 0
                    for i, mol in enumerate(mols):
                        selfie += fracs[i] * np.repeat(chemstructure[mol], data.shape[0]).reshape(52,data.shape[0]).T
                    # print(selfie.shape)
                    # print(data.shape)
                    temp_arr = np.full((len(data), 1), temp) 
                    data=np.hstack((selfie,temp_arr, data))
                    data_IAST.append(data)
    return np.vstack(data_RASPA),np.vstack(data_IAST)

chemstructure=ML_database()
max_amount_mols = 3
data_set_raspa, data_set_iast = make_training_database_ver2(max_amount_mols)
print(data_set_iast)