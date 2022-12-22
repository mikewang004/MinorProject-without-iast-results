# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version  
import tensorflow as tf


#making database of molecules representation
selfies_database = ML_database()
easy_database = simple_database()

#getting IAST combined with the molecule representaion
x_data, y_data = make_IAST_database_Wessel_version(easy_database,3)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size= 0.3)

