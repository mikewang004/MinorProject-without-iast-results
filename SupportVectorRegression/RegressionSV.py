# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:44:10 2022

@author: Wesse
"""

import sklearn as sk
from matplotlib import pyplot as plt
from selfies import data_gathering, ML_database
import pandas as pd

path_to_output = "Raspa/outputs"


data = data_gathering(path_to_output)
chemstructure = ML_database()

print(chemstructure)

for molecule in data:
    plt.title(molecule)
    plt.semilogx(data[molecule]["pressure"],data[molecule]["molkg"])
    plt.xlabel("pressure")
    plt.ylabel("mol/kg")
    plt.show()


