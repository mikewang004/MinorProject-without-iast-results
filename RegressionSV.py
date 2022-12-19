# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:44:10 2022

@author: Wesse
"""

from sklearn.model_selection import train_test_split  
import tensorflow as tf
from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version
import numpy as np


#making database of molecules representation
selfies_database = ML_database()
easy_database = simple_database()

#getting IAST combined with the molecule representaion
x_data, y_data = make_IAST_database_Wessel_version(easy_database,3)

#x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size= 0.3)

