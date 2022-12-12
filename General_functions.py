import selfies as sf
import numpy as np
# import os
# import pandas as pd
import glob

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

def simple_database():
    database = {}
    # array [c atomen, hoeveel braches, hoeveel c atomen in branches]
    database["C7"] = np.array([7,0,0])
    database["2mC6"] = np.array([7,1,1])
    database["3mC6"] = np.array([7,1,1])
    database['22mC5'] = np.array([7,2,1])
    database["23mC5"] = np.array([7,2,1])
    database['24mC5'] = np.array([7,2,1])
    database['33mC5'] = np.array([7,2,1])
    database["3eC5"] = np.array([7,1,2])
    database['223mC4'] = np.array([7,3,1])
    
    return database


# def data_gathering(path_to_output):#outdated function
#     data = {}
#     outputmaps = os.listdir(path_to_output)
#     for outputmap in outputmaps:
#         mappath = path_to_output + "/" + str(outputmap)
#         if os.path.isdir(mappath):
#             files = os.listdir(mappath)
#             for file in files:
#                 try:
#                     paths =  mappath + "/" + str(file)
#                     label = file.split("out")[0]
#                    #print(label)
#                     df = pd.read_table(paths, delimiter = ",")
#                     #df = df.set_index("pressure")
#                     data[label] = df.drop(["_","muc", "muc_err"], axis = 1)
#                     #print(data)
#                 except:
#                     print("ERROR !!!, please check " + file + " \n")
#     return data

def make_RASPA_database(chemstructure=ML_database()):
    path_to_out="Raspa/outputs/**/*.txt"
    data_RASPA=[]
    
    paths = glob.glob(path_to_out)
    for file in paths:
        molecule = file.split('/')[-1].split('-')[0]
        data=np.genfromtxt(file,delimiter=',',usecols=(0,3),skip_header=1)

        #Removing pressures that are too high
        data=np.delete(data,obj=np.where(data[:,0]>1e8),axis=0)
        # np.savetxt(file,data,delimiter=',',header='pressure,muc,muc_err,molkg,molkg_err,_')
        #adding temperature
        temp=int( file.split('/')[-1].split("out")[0][-3:] )
        data=np.insert(data,obj=1,axis=1,values=temp*np.ones(np.shape(data)[0]))
        
        #adding selfie
        selfie=np.repeat(chemstructure[molecule], data.shape[0]).reshape(52,data.shape[0]).T
        data=np.hstack((selfie,data))
        data_RASPA.append(data)
        
    return np.vstack(data_RASPA)
        
# def IAST_database():#outdated: uses old IAST outputs, only for 2 molecules
#     path_to_out='MachineLearning/Outputs_IAST'
#     paths = glob.glob(path_to_out + "/*.txt")
    
#     new_path="MachineLearning/Outputs_IAST/"
#     for file in paths:
#         #Removing pressures that are too high
#         data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float)
#         data=np.delete(data,obj=np.where(data[:,0]>1e8),axis=0)
        
#         length=np.ones(np.shape(data)[0])
#         file_split=file.split('-')
        
#         temp=int(file_split[1])
#         f1=float(file_split[2][:3])
#         f2=float(file_split[-1][:3])
        
#         data=np.insert(data,obj=1,axis=1,values=temp*length)
#         data=np.insert(data,obj=2,axis=1,values=f1*length)
#         data=np.insert(data,obj=3,axis=1,values=f2*length)
        
#         fname=file.split('/')[-1]
#         np.savetxt(new_path+fname, data,header='pressure,temperature,f1,f2,molkg1,molkg2',delimiter=',')

def make_IAST_database(chemstructure=ML_database):
    
    path_IAST=glob.glob('IAST-segregated/automated_output/**/**/**/*.txt')
    chemstructure=ML_database()
    data_2=[]
    data_3=[]
    data_4=[]
    data_5=[]
    for file in path_IAST: 
        file = file.replace('\\','/')
        folders=file.split('/')
        temp=int( folders[3][:3] )
        molnum=int(folders[2][0] )
        data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float)
        data=np.insert(data,obj=1,axis=1,values=temp)
        
        if molnum==2:
            m1=folders[4].split("-")[0]
            m2=folders[4].split("-")[1]
            
            f1=float( folders[-1].split('-')[0] )
            f2=1-f1
            
            selfie=f1*chemstructure[m1]+f2*chemstructure[m2]
            selfie=np.repeat(selfie, data.shape[0]).reshape(52,data.shape[0]).T
            
            data=np.hstack((selfie,data))
            data_2.append(data)
        elif molnum==3:
            m1=folders[4].split("-")[0]
            m2=folders[4].split("-")[1]
            m3=folders[4].split("-")[2]
            
            f1=float( folders[-1].split('-')[0])
            f2=float( folders[-1].split('-')[1])
            f3=1-(f1+f2)
            
            selfie=f1*chemstructure[m1]+f2*chemstructure[m2]+f3*chemstructure[m3]
            selfie=np.repeat(selfie, data.shape[0]).reshape(52,data.shape[0]).T
            
            data=np.hstack((selfie,data))
            data_3.append(data)
        elif molnum==4:
            m1=folders[4].split("-")[0]
            m2=folders[4].split("-")[1]
            m3=folders[4].split("-")[2]
            m4=folders[4].split("-")[3]
            
            f1=float( folders[-1].split('-')[0])
            f2=float( folders[-1].split('-')[1])
            f3=float( folders[-1].split('-')[2])
            f4=1-(f1+f2+f3)
            
            selfie=f1*chemstructure[m1]+f2*chemstructure[m2]+f3*chemstructure[m3]+f4*chemstructure[m4]
            selfie=np.repeat(selfie, data.shape[0]).reshape(52,data.shape[0]).T
            
            data=np.hstack((selfie,data))
            data_4.append(data)
        elif molnum==5:
    
            m1=folders[4].split("-")[0]
            m2=folders[4].split("-")[1]
            m3=folders[4].split("-")[2]
            m4=folders[4].split("-")[3]
            m5=folders[4].split("-")[4]
            
            f1=float( folders[-1].split('-')[0])
            f2=float( folders[-1].split('-')[1])
            f3=float( folders[-1].split('-')[2])
            f4=float( folders[-1].split('-')[3])
            f5=1-(f1+f2+f3+f4)
            
            selfie=f1*chemstructure[m1]+f2*chemstructure[m2]+f3*chemstructure[m3]+f4*chemstructure[m4]+f5*chemstructure[m5]
            selfie=np.repeat(selfie, data.shape[0]).reshape(52,data.shape[0]).T
            
            data=np.hstack((selfie,data))
            data_5.append(data)
    return np.vstack(data_2),np.vstack(data_3),np.vstack(data_4),np.vstack(data_5)


def make_IAST_database_ver2(nummol, chemstructure=ML_database):
    """
    Enter the number of molecules you want to mix (nummol). The function will
    in default use the ML_database for the chemical structure of the molecules, 
    which is a dictionary containing all selfie structures of the used 
    molecules.
    The function will return the x and y values of the IAST-data:
     * The variable x_vals is an 2D array which contains of multiple arrays 
       filled with the following: fraction of mixture molecule
       1, selfies code molecule 1, ..., fraction of mixture molecule nummol, 
       selfies code molecule nummol, the pressure, the temprature. In each 
       array (inside x_vals) is either the pressure or temprature is varied.
     * The variable y_vals contains the following: loading of molecule 1, ...,
       loading of molecule nummol. 
       
    With the return values of this function you can create a train and test
    data set for your machine learning algorithm, using the following code 
    snippet:
        
    x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals,
                                            test_size= 0.1, random_state=0)  
    """
    #create list of all paths to the data files of the mixture of nummol molecules:
    path_IAST=glob.glob(f'IAST-segregated/automated_output/{nummol}_molecules/**/**/*.txt')
    
    chemstructure=ML_database()
    data_IAST =[]
    for file in path_IAST: 
        file = file.replace("\\", "/") #comment this line if you use linux
        folders=file.split('/') #creates a list of all folders in the directory leading to the file
        temp=int( folders[3][:3] ) #extracts the temperature of the mixture from the folders list
        
        #load data and insert the temperature at the end of each row of data:
        data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float) 
        data=np.insert(data,obj=1,axis=1,values=temp)
        
        for molnum in range(nummol):
            #create an array of the fraction (frac) and selfie (selfie) of 
            #molecule molnum and add it to the array frac_selfie, where the two
            #are combined. After the for loop the array frac_selfie contains
            #of rows filled with the fraction of molecule 1, selfie of molecule
            #1, ..., fraction of molecule nummol, selfie of nummol. frac_selfie
            #contains of the same amount of columns as the data array and each
            #row is identical.
            frac =  np.full((len(data), 1), float(folders[-1].split('-')[molnum].split(".txt")[0]))
            selfie = np.repeat(chemstructure[folders[4].split("-")[molnum]], data.shape[0]).reshape(52,data.shape[0]).T
            if molnum == 0:
                frac_selfie = np.hstack((frac, selfie))
            else:
                frac_selfie = np.hstack((frac_selfie, frac, selfie))
        
        #combine the frac_selfie array and the data array into one array. And 
        #append the combined array to data_IAST, which will gather all combined 
        #arrays:
        data=np.hstack((frac_selfie,data)) 
        data_IAST.append(data)
    
    data_IAST = np.vstack(data_IAST) #reorder the array
    
    #seperate the input values (x_vals) and output values (y_vals) from the 
    #combined data set (data_IAST):
    x_vals=data_IAST[:,:-nummol] 
    y_vals=data_IAST[:,(len(data_IAST[0]))-nummol:]    
    return x_vals, y_vals

def make_IAST_database_Wessel_version(chemstructure,n_molecule_combinations, only_max_combinations = False,):
    
    """
    Generate inputvectordata and outputvectordata 
    
    :param chemstructure: database which contains the molecule representation
    :param n_molecule_combinations: maximum number of different molecules in a single mixture
    :param only_max_combinations: 
        -if False every different molecule mixture up to the max different molecules will be put in the dataset
        -if true: only the max different molecule mixtures will be included into the dataset
        
    :return the inputvector_data: the inputvectors from the dataset which is structured the following:
            [molfraction1,chemstructure1, molfraction2, chemstructure2,...,molfractionN, chemstructureN]
    :return the outputvector_data: the inputvectors from the dataset which is structured the following:
             [loadingmolecule1,loadingmolecul2,..,loadingmoleculeN]
    """
    
    path_IAST=glob.glob('IAST-segregated/automated_output/**/**/**/*.txt') #looking for all the IAST files
    
    Length_inputvector = (len(chemstructure["C7"]) +1 ) * n_molecule_combinations #determining the lenght of the inputvector
    total_data = []
    
    "going over every file in the directory " 
    for file in path_IAST: 
        file = file.replace('\\','/') #"linux conversion stuff"
        folders=file.split('/')
        
        temperature=int( folders[3][:3] ) # getting the temperature
        molnum=int(folders[2][0] ) #getting the number of n mixure molecules
        
        """checking of the number of molecules in the mixture is larger than what we want 
            if so than skip this file"""
        
        if only_max_combinations:
            if molnum != n_molecule_combinations:
                continue
            
        if molnum > n_molecule_combinations: 
            continue
        
        """loading in data"""
        data=np.genfromtxt(file,delimiter='    ',skip_header=1,dtype=float) 
        data=np.insert(data,obj=1,axis=1,values=temperature) #inserting the temperatyr 
        
        """getting molecule names and mixture ratios from the paths"""
        moleculenames = folders[4].split("-") #list of the molecule names
        mixtureratios = folders[-1].replace(".txt","").split("-")   #getting mixture ratios  from foldername 
        
        
        """creating the input vector"""
        inputvector = np.array([])
        for mixtureratio , molecule in zip( mixtureratios , moleculenames):
            inputvector = np.hstack((inputvector,float(mixtureratio),chemstructure[molecule]))

        """making sure that the array is correct size and otherwise adding zeros until it is the desired length"""      
        checklength = len(inputvector)
        if checklength < Length_inputvector :
            inputvector = np.hstack((inputvector, np.zeros( ( Length_inputvector - checklength)) ) )
      
        """Adding the chemstructure to each pressure """
        inputvector =np.repeat(inputvector, data.shape[0]).reshape(Length_inputvector ,data.shape[0]).T
        
        data=np.hstack((inputvector,data))
        total_data.append(data)
           
    total_data = np.vstack(total_data) #stacking vertically so that it becomes a 2D array instead of 3D array
        

    return total_data[:,:(Length_inputvector+2)], total_data[:,Length_inputvector+2:]