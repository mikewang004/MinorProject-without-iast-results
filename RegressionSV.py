# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:44:10 2022

@author: Wesse
"""

import sklearn as sk
from matplotlib import pyplot as plt
from General_functions import ML_database, data_gathering, simple_database
import pandas as pd
import numpy as np


path_to_output = "Raspa/outputs"

#collecting all data
data = data_gathering(path_to_output)
selfies_database = ML_database()
easy_database = simple_database()

molecule1 = "C7"
T1 = 300

molecule2 = "23mC5"
T2 = 400

