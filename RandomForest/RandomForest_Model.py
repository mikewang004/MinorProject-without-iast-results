import matplotlib.pyplot as plt
import numpy as np
from joblib import load


def XMolMix_dataset(amount_mols):
    data = np.load(f"data_{amount_mols}molsmix.npy", allow_pickle = True)
    x_train = data[0]
    x_test = data[1]
    y_train = data[2] 
    y_test = data[3]
    return x_train, x_test, y_train, y_test
    
def XMolMix_model(amount_mols):
    return load(f'rf_{amount_mols}mols_mix.joblib')

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
    mean_abs_err = np.mean(abs_err)
    
    plt.figure()
    plt.title(f"Relative error Decision Tree, {amount_mols} molecules micture")
    plt.scatter(range(len(rel_err)), rel_err, label="Relative error point i")
    plt.hlines(mean_rel_err, xmin = 0, xmax = len(rel_err), color="red", label=f"Mean relative error = {round(mean_rel_err,5)}")
    plt.yscale("log")
    plt.xlabel("Index of array rel_err")
    plt.ylabel("Relative error of predicted point wrt to known point")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(f"Absolute error Decision Tree, {amount_mols} molecules micture")
    plt.scatter(range(len(abs_err)), abs_err, label="Absolute error point i")
    plt.hlines(mean_abs_err, xmin = 0, xmax = len(abs_err), color="red", label=f"Mean absolute error = {round(mean_abs_err,5)}")
    plt.yscale("log")
    plt.xlabel("Index of array abs_err")
    plt.ylabel("Absolute error of predicted point wrt to known point")
    plt.legend()
    
    plt.figure()
    plt.title(f"Performance Decision Tree, {amount_mols} molecules micture")
    plt.scatter(y_test, y_pred)
    plt.xlabel("True loading (from IAST)")
    plt.ylabel("Predicted loading (from Desicion Tree)")
    plt.show()
    
    print(f"Mean relative error = {mean_rel_err}")
    print(f"Mean absolute error = {mean_abs_err}")


x_train, x_test, y_train, y_test = XMolMix_dataset(2)
two_mol_mix_rf = XMolMix_model(2)
Performance(2, two_mol_mix_rf, x_train, x_test, y_train, y_test)











