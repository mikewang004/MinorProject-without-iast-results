import os

os.chdir("..")

from sklearn.ensemble import RandomForestRegressor
from General_functions import make_IAST_database_ver2, simple_database
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt
import numpy as np
import time

def Performance(amount_mols, rf_model, x_train, x_test, y_train, y_test):
    y_pred=rf_model.predict(x_test) 
    
    
    abs_err = np.abs(y_pred-y_test)
    rel_err = abs_err/y_test
    
    nanIndex = np.isnan(rel_err)
    rel_err = rel_err[~nanIndex] #to remove nan's
    infIndex = np.isinf(rel_err)
    rel_err = rel_err[~infIndex] #to remove zero's
    
    abs_err = abs_err[~nanIndex] #to remove nan's
    abs_err = abs_err[~infIndex] #to remove zero's
    
    rel_err = rel_err.flatten()
    abs_err = abs_err.flatten()
    
    mean_rel_err = np.mean(rel_err)
    std_rel_err = np.std(rel_err)
    mean_abs_err = np.mean(abs_err)
    std_abs_err = np.std(abs_err)

    
    # plt.figure()
    # plt.title(f"Relative error Decision Tree, {amount_mols} molecules micture")
    # plt.scatter(range(len(rel_err)), rel_err, label="Relative error point i")
    # plt.hlines(mean_rel_err, xmin = 0, xmax = len(rel_err), color="red", label=f"Mean relative error = {round(mean_rel_err,5)}")
    # plt.yscale("log")
    # plt.xlabel("Index of array rel_err")
    # plt.ylabel("Relative error of predicted point wrt to known point")
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.title(f"Absolute error Decision Tree, {amount_mols} molecules micture")
    # plt.scatter(range(len(abs_err)), abs_err, label="Absolute error point i")
    # plt.hlines(mean_abs_err, xmin = 0, xmax = len(abs_err), color="red", label=f"Mean absolute error = {round(mean_abs_err,5)}")
    # plt.yscale("log")
    # plt.xlabel("Index of array abs_err")
    # plt.ylabel("Absolute error of predicted point wrt to known point")
    # plt.legend()
    
    plt.figure()
    plt.title(f"Performance Decision Tree, {amount_mols} molecules micture")
    plt.scatter(y_test, y_pred)
    plt.xlabel("True loading (from IAST)")
    plt.ylabel("Predicted loading (from Desicion Tree)")
    plt.show()
    
    print(f"Mean relative error = {mean_rel_err}")
    print(f"Mean absolute error = {mean_abs_err}")
    
    return  mean_rel_err,std_rel_err,mean_abs_err,std_abs_err


amount_mols = 4
from sklearn.neural_network import MLPRegressor

x_vals, y_vals = make_IAST_database_ver2(amount_mols, simple_database())
x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals,
                                        test_size=0.1, random_state=0)  

# abs_err_arr = []
# rel_err_arr = []
# std_rel_err_array =[]
# std_abs_err_array =[]

# for i in range(1,150):
#     regr = RandomForestRegressor(n_estimators=i, random_state=0,n_jobs=-1)#max_samples,n_jobs=-1
#     regr.fit(x_train, y_train)
#     mean_rel_err,std_rel_err,mean_abs_err,std_abs_err = Performance(amount_mols, regr, x_train, x_test, y_train, y_test)
#     abs_err_arr=np.append(abs_err_arr, mean_abs_err)
#     rel_err_arr=np.append(rel_err_arr, mean_rel_err)
#     std_rel_err_array=np.append(std_rel_err_array, std_rel_err)
#     std_abs_err_array=np.append(std_abs_err_array, std_abs_err)
#     print(i)
    
#     plt.figure()
#     plt.title("Relative error Decision Tree, various number of trees")
#     plt.scatter(range(1,i+1), rel_err_arr, label="Relative error point i")
#     # plt.errorbar(range(1,i+1), rel_err_arr, x_err=None, y_err=std_rel_err_array)
#     plt.xlabel("Number of trees")
#     plt.ylabel("Mean relative error")
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.title("Absolute error Decision Tree, various number of trees")
#     plt.scatter(range(1,i+1), abs_err_arr, label="Relative error point i")
#     # plt.errorbar(range(1,i+1), abs_err_arr, x_err=None, y_err=std_abs_err_array)
#     plt.xlabel("Number of trees")
#     plt.ylabel("Mean relative error")
#     plt.legend()
#     plt.show()
# from joblib import dump, load
# os.chdir("RandomForest")
# np.save("data_4molsmix.npy", np.array([x_train, x_test, y_train, y_test]))
# dump(regr, 'rf_4mols_mix.joblib') 

start = time.time()
regr = MLPRegressor(random_state=0)
# regr = RandomForestRegressor(n_estimators=100, random_state=0,n_jobs=-1)#max_samples,n_jobs=-1
regr.fit(x_train, y_train)
end = time.time()
print(end-start)
mean_rel_err,std_rel_err,mean_abs_err,std_abs_err = Performance(amount_mols, regr, x_train, x_test, y_train, y_test)


