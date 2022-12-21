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
from General_functions import ML_database ,simple_database, make_IAST_database_Wessel_version
#Make the database of the molecule representation
selfies_database = ML_database()
easy_database = simple_database()

x_data, y_data = make_IAST_database_Wessel_version(easy_database,3)